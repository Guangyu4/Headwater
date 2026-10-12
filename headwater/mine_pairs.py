"""PMPO preference pairs mined from the labels and the swept-along map, without annotation.

Every pair keeps the prompt of one faulty trace. The chosen answer is the label, and the
rejected answer, the decoy, changes exactly one field:

  source_peer           the source becomes the earliest swept-along peer, ties broken by hop
                        distance to the source and then by id
  onset_peer            the onset becomes the round at which that peer was reached
  pathology_confusable  the pathology becomes the class that the SFT model confuses most often
                        with the true one, read from the `pathology_confusion` block of a
                        headwater.evaluate report on the validation split

`reward_gap` = r(y) - r(y-) under headwater/reward.py is the reward that the decoy forfeits,
and run_pmpo scales it by --margin-mu into the target margin.

Run:  python -m headwater.mine_pairs --traces train.jsonl --confusion-report eval/sft_val_report.json --out pairs.parquet
"""

import argparse
import json
import os
from collections import deque
from typing import Dict, List, Optional

import pandas as pd

from .common import answer_json, build_prompt, read_jsonl
from .reward import PATHOLOGIES, compute_score

# look-alike classes used when no confusion report is given
CONFUSABLE_PRIOR: Dict[str, str] = {
    "conformity": "sycophancy",
    "sycophancy": "conformity",
    "persona_drift": "memory_inconsistency",
    "memory_inconsistency": "persona_drift",
    "relay_distortion": "memory_inconsistency",
    "stereotyping": "persona_drift",
}
PAIR_TYPES = ("source_peer", "onset_peer", "pathology_confusable")


def parse_edges(input_text: str) -> Dict[str, List[str]]:
    """Adjacency from the NETWORK EDGES block of the record."""
    marker = "=== NETWORK EDGES ==="
    start = input_text.find(marker)
    if start < 0:
        return {}
    body = input_text[start + len(marker):].split("===", 1)[0]
    graph: Dict[str, List[str]] = {}
    for token in body.replace("\n", " ").split(","):
        token = token.strip()
        if "-" not in token:
            continue
        a, b = token.split("-", 1)
        graph.setdefault(a.strip(), []).append(b.strip())
        graph.setdefault(b.strip(), []).append(a.strip())
    return graph


def hops(graph: Dict[str, List[str]], src: str, dst: str) -> int:
    if src == dst:
        return 0
    seen, queue = {src}, deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt == dst:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return -1


def earliest_peer(sample: dict, graph: Dict[str, List[str]]) -> Optional[tuple]:
    """(peer, round reached) of the earliest swept-along peer."""
    src = sample["answer"]["source"]
    peers = [(r, hops(graph, src, a) if graph else 0, a)
             for a, r in (sample.get("swept_along") or {}).items() if a != src]
    if not peers:
        return None
    r, _, a = sorted(peers, key=lambda t: (t[0], t[1] if t[1] >= 0 else 99, t[2]))[0]
    return a, int(r)


def pairs_for(sample: dict, confusable: Dict[str, str], types: List[str]) -> List[dict]:
    truth = sample["answer"]
    extra = {"rounds": sample["rounds"], "agents": sample["agents"],
             "swept_along_json": json.dumps(sample.get("swept_along") or {})}
    truth_json = answer_json(truth)
    truth_reward = compute_score("", truth_json, truth_json, extra)["score"]
    out: List[dict] = []

    def emit(decoy_type: str, rejected: dict):
        rejected_json = answer_json(rejected)
        gap = truth_reward - compute_score("", rejected_json, truth_json, extra)["score"]
        out.append({"sample_id": sample["sample_id"], "scenario": sample["scenario"],
                    "decoy_type": decoy_type, "reward_gap": round(gap, 6),
                    "chosen_answer": truth_json, "rejected_answer": rejected_json})

    peer = earliest_peer(sample, parse_edges(sample["input"]))
    if peer is not None:
        peer_id, peer_round = peer
        if "source_peer" in types:
            emit("source_peer", {**truth, "source": peer_id})
        if "onset_peer" in types and peer_round != truth["onset"]:
            emit("onset_peer", {**truth, "onset": peer_round})
    if "pathology_confusable" in types and truth["pathology"] in confusable:
        emit("pathology_confusable", {**truth, "pathology": confusable[truth["pathology"]]})
    return out


def to_trl_rows(pairs: List[dict], inputs: Dict[str, str]) -> List[dict]:
    """TRL conversational preference format, with the pair metadata alongside."""
    return [{"prompt": [{"role": "user", "content": build_prompt(inputs[p["sample_id"]])}],
             "chosen": [{"role": "assistant", "content": p["chosen_answer"]}],
             "rejected": [{"role": "assistant", "content": p["rejected_answer"]}],
             **{k: v for k, v in p.items() if k not in ("chosen_answer", "rejected_answer")}}
            for p in pairs]


def confusable_from_report(path: str) -> Dict[str, str]:
    """Per true class, the most frequent wrong prediction of the report's model, falling back
    to the prior for classes it never confuses."""
    with open(path, encoding="utf-8") as f:
        matrix = json.load(f)["pathology_confusion"]
    table = dict(CONFUSABLE_PRIOR)
    for true_cls, preds in matrix.items():
        wrong = {k: v for k, v in preds.items() if k != true_cls and k in PATHOLOGIES and v > 0}
        if wrong:
            table[true_cls] = max(sorted(wrong), key=lambda k: wrong[k])
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="trace JSONL of the training split")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--confusion-report", default=None, help="evaluate report of the SFT model on validation")
    ap.add_argument("--types", nargs="+", default=list(PAIR_TYPES), choices=PAIR_TYPES)
    args = ap.parse_args()

    confusable = confusable_from_report(args.confusion_report) if args.confusion_report else CONFUSABLE_PRIOR
    samples = list(read_jsonl(args.traces))
    pairs = [p for s in samples for p in pairs_for(s, confusable, args.types)]
    rows = to_trl_rows(pairs, {s["sample_id"]: s["input"] for s in samples})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    counts = {t: sum(r["decoy_type"] == t for r in rows) for t in args.types}
    print(json.dumps({"pairs": len(rows), "by_type": counts, "confusable": confusable}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
