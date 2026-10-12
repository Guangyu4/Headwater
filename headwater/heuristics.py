"""Rule-based auditor baselines that answer from the serialized trace alone, without a model.

  random     a uniformly random agent, pathology, and onset (seeded)
  talkative  the agent with the most characters in the shown turns, onset at the first
             round, and the most frequent training pathology
  deviant    the (agent, round) with the largest jump of the agent's declared value from its
             own previous value, onset at that round, conformity when the jump moved toward
             the mean of the other agents and memory inconsistency otherwise (numeric scales
             are normalized to [0, 1], and a categorical change counts as 1)

The output is a predictions JSONL ({"sample_id", "output"}) for
    python -m headwater.evaluate --data <same file> --predictions <out> --out <prefix>

Run:  python -m headwater.heuristics --data data/headwater_bench_sample.jsonl --strategy deviant --out eval/deviant.jsonl
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .common import write_jsonl
from .evaluate import load_items
from .reward import PATHOLOGIES

STRATEGIES = ("random", "talkative", "deviant")
PRIOR_PATHOLOGY = "conformity"
TRANSCRIPT_LINE = re.compile(r"^\[r(\d+)\]\s+(\S+):\s*(.*)$")
TAG = re.compile(r"<(stance|adopt|contribute|answer)>\s*([^<]*?)\s*</\1>", re.IGNORECASE)
SCALE = {"stance": 100.0, "adopt": 1.0}
VALUE_TABLE_LINE = re.compile(r"^\[r(\d+)\]\s+(.*)$")
VALUE_CELL = re.compile(r"(\S+)=(\S+)")


def transcript_of(record_text: str) -> List[Tuple[int, str, str]]:
    """(round, agent, utterance) rows of the TRANSCRIPT or TRANSCRIPT EXCERPTS block."""
    for header in ("=== TRANSCRIPT ===", "=== TRANSCRIPT EXCERPTS ==="):
        block = record_text.split(header, 1)
        if len(block) == 2:
            break
    else:
        return []
    body = block[1].split("\n===", 1)[0]
    rows = []
    for line in body.strip().splitlines():
        m = TRANSCRIPT_LINE.match(line.strip())
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3)))
    return rows


def declared_values_of(record_text: str) -> List[Tuple[int, str, str]]:
    """(round, agent, value) cells of the DECLARED VALUES table, skipping '?' cells."""
    block = record_text.split("=== DECLARED VALUES BY ROUND ===", 1)
    if len(block) < 2:
        return []
    body = block[1].split("\n===", 1)[0]
    rows = []
    for line in body.strip().splitlines():
        m = VALUE_TABLE_LINE.match(line.strip())
        if not m:
            continue
        rnd = int(m.group(1))
        for agent, value in VALUE_CELL.findall(m.group(2)):
            if value != "?":
                rows.append((rnd, agent, value))
    return rows


def tagged_value(utterance: str) -> Tuple[Optional[str], Optional[str]]:
    """The last scenario tag of an utterance as (tag, raw value), or (None, None)."""
    found = TAG.findall(utterance)
    if not found:
        return None, None
    tag, value = found[-1]
    return tag.lower(), value.strip()


def series_of(record_text: str) -> Dict[str, Dict[int, Tuple[str, str]]]:
    """agent -> round -> (tag, value), from the value table when the record has one and from
    the tags of the shown turns otherwise."""
    series: Dict[str, Dict[int, Tuple[str, str]]] = defaultdict(dict)
    table = declared_values_of(record_text)
    if table:
        for rnd, agent, value in table:
            series[agent][rnd] = ("value", value)
        return series
    for rnd, agent, text in transcript_of(record_text):
        tag, value = tagged_value(text)
        if tag is not None:
            series[agent][rnd] = (tag, value)
    return series


def _numeric(tag: str, value: str, value_max: float) -> Optional[float]:
    try:
        x = float(value)
    except ValueError:
        return None
    if tag in ("contribute", "value"):
        return x / value_max if value_max > 0 else None
    return x / SCALE.get(tag, 1.0)


def deviant_answer(series: Dict[str, Dict[int, Tuple[str, str]]]) -> dict:
    values = []
    for by_round in series.values():
        for tag, value in by_round.values():
            if tag in ("contribute", "value"):
                try:
                    values.append(abs(float(value)))
                except ValueError:
                    pass
    value_max = max(values) if values else 0.0
    numeric = {(agent, rnd): (_numeric(tag, value, value_max) if tag != "answer" else None)
               for agent, by_round in series.items() for rnd, (tag, value) in by_round.items()}

    best = (0.0, None, None, None)   # (jump, agent, round, moved toward the mean)
    for agent, by_round in series.items():
        rounds = sorted(by_round)
        for prev, cur in zip(rounds, rounds[1:]):
            a, b = numeric.get((agent, prev)), numeric.get((agent, cur))
            if a is None or b is None:
                jump, toward = (0.0 if by_round[prev] == by_round[cur] else 1.0), None
            else:
                jump = abs(b - a)
                peers = [v for (ag, r), v in numeric.items() if r == cur and ag != agent and v is not None]
                mean = sum(peers) / len(peers) if peers else None
                toward = None if mean is None else abs(b - mean) < abs(a - mean)
            if jump > best[0]:
                best = (jump, agent, cur, toward)
    jump, agent, rnd, toward = best
    if agent is None:
        return {"source": None, "pathology": None, "onset": None}
    return {"source": agent, "pathology": "conformity" if toward else "memory_inconsistency", "onset": rnd}


def talkative_answer(rows: List[Tuple[int, str, str]]) -> dict:
    chars: Dict[str, int] = defaultdict(int)
    first_round: Dict[str, int] = {}
    for rnd, agent, text in rows:
        chars[agent] += len(text)
        first_round.setdefault(agent, rnd)
    if not chars:
        return {"source": None, "pathology": None, "onset": None}
    agent = max(sorted(chars), key=lambda a: chars[a])
    return {"source": agent, "pathology": PRIOR_PATHOLOGY, "onset": min(first_round.values())}


def predict(item: dict, strategy: str, rng: random.Random) -> dict:
    record = "\n\n".join(m["content"] for m in item["messages"])
    if strategy == "random":
        extra = item["extra_info"]
        return {"source": rng.choice(sorted(extra["agents"])), "pathology": rng.choice(PATHOLOGIES),
                "onset": rng.randrange(max(1, int(extra["rounds"])))}
    if strategy == "talkative":
        return talkative_answer(transcript_of(record))
    if strategy == "deviant":
        return deviant_answer(series_of(record))
    raise ValueError(f"unknown strategy {strategy!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="trace JSONL or rl_*.parquet")
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--out", required=True, help="predictions JSONL")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random strategy")
    args = ap.parse_args()
    items = load_items(args.data)
    rng = random.Random(args.seed)
    rows = [{"sample_id": it["sample_id"], "output": json.dumps(predict(it, args.strategy, rng))} for it in items]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{args.strategy}: {write_jsonl(args.out, rows)} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
