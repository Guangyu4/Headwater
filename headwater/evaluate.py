"""Field-wise evaluation of an auditor on faulty traces.

Every auditor gets the attribution prompt, which states that exactly one participant
deviated. Reported per group, all over faulty traces with a malformed or all-null answer
counting as wrong:
  source_acc, pathology_acc, onset_acc   exact-match accuracies, the paper's metrics
  onset_within_1, onset_mae              onset error, the MAE over reported answers only
  swept_along_error_rate                 a wrong source that is a peer the fault swept along
  format_fail_rate, null_rate, mean_reward
Groups are all traces, each scenario, pathology, network, injection surface, source
position, and onset tier, and `scenario_average` is the unweighted mean over scenarios.
`pathology_confusion` feeds the pathology decoys of headwater.mine_pairs.

Backends: `vllm` (a Hugging Face checkpoint), `hf` (transformers generate, optionally
with a PEFT adapter), and `api` (an OpenAI-compatible endpoint set by OPENAI_BASE_URL and
OPENAI_API_KEY). The api backend appends answers to <out>_raw.jsonl as they arrive, so a
stopped run resumes where it stopped. --predictions scores an existing
{"sample_id", "output"} JSONL instead.

Run:  python -m headwater.evaluate --data data/headwater_bench_sample.jsonl --model <checkpoint> --out eval/bench
      python -m headwater.evaluate --data data/headwater_bench_sample.jsonl --backend api --model gpt-5.6-sol --out eval/gpt
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .common import DATA_SOURCE, assume_fault, build_prompt, read_jsonl, render_chat, user_message, write_jsonl
from .reward import PATHOLOGIES, classify, compute_score, extract_answer

FORMAT_REMINDER = (
    "Answer again with a single JSON object and nothing else, exactly in this form:\n"
    '{"source": "<agent id>", "pathology": "<one of the six names>", "onset": <round index>}'
)
SLICE_AXES = ("network", "surface", "position")


# ---- data ----

def load_items(path: str) -> List[dict]:
    """Items {sample_id, messages, ground_truth, extra_info, scenario} from a trace JSONL or
    an rl_*.parquet of headwater.to_parquet, with the attribution prompt."""
    items = []
    if path.endswith(".parquet"):
        import pandas as pd
        for _, row in pd.read_parquet(path).iterrows():
            extra = dict(row["extra_info"])
            if hasattr(extra.get("agents"), "tolist"):
                extra["agents"] = extra["agents"].tolist()
            items.append({"sample_id": extra["sample_id"], "messages": [dict(m) for m in row["prompt"]],
                          "ground_truth": row["reward_model"]["ground_truth"], "extra_info": extra,
                          "scenario": extra["scenario"]})
    else:
        for s in read_jsonl(path):
            extra = {"rounds": s["rounds"], "agents": s["agents"],
                     "swept_along_json": json.dumps(s.get("swept_along") or {}),
                     **{axis: s.get(axis) for axis in SLICE_AXES}}
            items.append({"sample_id": s["sample_id"], "messages": user_message(build_prompt(s["input"])),
                          "ground_truth": json.dumps(s["answer"]), "extra_info": extra,
                          "scenario": s["scenario"]})
    for it in items:
        it["messages"] = [dict(m, content=assume_fault(m["content"])) if m["role"] == "user" else m
                          for m in it["messages"]]
    return items


def onset_tier(onset: int, rounds: int) -> str:
    """Early, mid, or late third of the trace."""
    frac = onset / max(1, rounds - 1)
    return "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")


# ---- generation ----

def generate_vllm(model: str, items: List[dict], max_new_tokens: int, tp: int, max_model_len: int) -> List[str]:
    """Greedy decoding. Prompts are rendered by render_chat and tokenized without special
    tokens, since the template already carries them, so evaluation sees the token ids that
    training saw."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = [TokensPrompt(prompt_token_ids=tok(render_chat(tok, it["messages"]), add_special_tokens=False)["input_ids"])
               for it in items]
    llm = LLM(model=model, tensor_parallel_size=tp, trust_remote_code=True,
              enable_prefix_caching=True, max_model_len=max_model_len)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
    return [o.outputs[0].text for o in outs]


def generate_hf(model: str, items: List[dict], max_new_tokens: int, lora: Optional[str],
                batch_size: int) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto",
                                               trust_remote_code=True)
    if lora:
        from peft import PeftModel
        net = PeftModel.from_pretrained(net, lora)
    net.eval()
    texts = [render_chat(tok, it["messages"]) for it in items]
    outputs: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(net.device)
        with torch.no_grad():
            gen = net.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for row in gen[:, batch["input_ids"].shape[1]:]:
            outputs.append(tok.decode(row, skip_special_tokens=True))
    return outputs


def generate_api(client, items: List[dict], concurrency: int, cache_path: str) -> List[str]:
    """Zero-shot baseline through an OpenAI-compatible endpoint. An answer that fails the
    format gate is asked once more with FORMAT_REMINDER appended, and a second failure
    scores as wrong. Answers already in `cache_path` are not asked again."""
    from concurrent.futures import ThreadPoolExecutor
    done: Dict[str, str] = {}
    if os.path.exists(cache_path):
        done = {r["sample_id"]: r["output"] for r in read_jsonl(cache_path)}
    todo = [it for it in items if it["sample_id"] not in done]
    print(f"[api] {len(done)} cached, {len(todo)} to generate with {client.model}", flush=True)

    def one(it: dict):
        prompt = "\n\n".join(m["content"] for m in it["messages"])
        first = client.complete(prompt)
        extra = it["extra_info"]
        if classify(extract_answer(first), extra.get("agents"), extra.get("rounds"))[0] != "malformed":
            return it["sample_id"], first, None
        return it["sample_id"], client.complete(f"{prompt}\n\n{FORMAT_REMINDER}"), first

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as cache, ThreadPoolExecutor(max(1, concurrency)) as pool:
        for sample_id, text, first in pool.map(one, todo):
            done[sample_id] = text
            row = {"sample_id": sample_id, "output": text}
            if first is not None:
                row["first_output"] = first
            cache.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache.flush()
    print(f"[api] usage {client.usage}", flush=True)
    return [done[it["sample_id"]] for it in items]


# ---- scoring ----

def score_predictions(items: List[dict], outputs: Iterable[str]) -> dict:
    groups: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    confusion = {p: defaultdict(int) for p in PATHOLOGIES}
    rows = []
    for it, text in zip(items, outputs):
        extra = it["extra_info"]
        s = compute_score(DATA_SOURCE, text, it["ground_truth"], extra)
        truth = json.loads(it["ground_truth"])
        status, pred = classify(extract_answer(text), extra.get("agents"), extra.get("rounds"))
        if status == "report":
            confusion[truth["pathology"]][pred["pathology"]] += 1
        keys = ["all", f"scenario:{it['scenario']}", f"pathology:{truth['pathology']}",
                f"onset_tier:{onset_tier(int(truth['onset']), int(extra['rounds']))}"]
        keys += [f"{axis}:{extra[axis]}" for axis in SLICE_AXES if extra.get(axis)]
        for key in keys:
            g = groups[key]
            g["n"] += 1
            g["format_fail"] += 1 - s["format_ok"]
            g["null"] += s["null"]
            g["reward"] += s["score"]
            g["source_hit"] += s["source_hit"]
            g["pathology_hit"] += s["pathology_hit"]
            g["swept_along_error"] += s["swept_along_error"]
            if s["attributed"]:
                g["n_attributed"] += 1
                g["onset_abs_err"] += s["onset_abs_err"]
                g["onset_hit"] += float(s["onset_abs_err"] == 0)
                g["onset_within_1"] += float(s["onset_abs_err"] <= 1)
        rows.append({"sample_id": it["sample_id"], "output": text, "ground_truth": it["ground_truth"],
                     "pred": pred if status != "malformed" else None, "status": status, **s})

    def finish(g: Dict[str, float]) -> dict:
        n, na = g["n"], g["n_attributed"]
        return {
            "n": int(n),
            "source_acc": g["source_hit"] / n,
            "pathology_acc": g["pathology_hit"] / n,
            "onset_acc": g["onset_hit"] / n,
            "onset_within_1": g["onset_within_1"] / n,
            "onset_mae": g["onset_abs_err"] / na if na else None,
            "swept_along_error_rate": g["swept_along_error"] / n,
            "format_fail_rate": g["format_fail"] / n,
            "null_rate": g["null"] / n,
            "mean_reward": g["reward"] / n,
        }

    report = {key: finish(g) for key, g in sorted(groups.items())}
    scenarios = [v for k, v in report.items() if k.startswith("scenario:")]
    report["scenario_average"] = {m: sum(s[m] for s in scenarios) / len(scenarios)
                                  for m in ("source_acc", "pathology_acc", "onset_acc")}
    report["pathology_confusion"] = {k: dict(v) for k, v in confusion.items()}
    return {"report": report, "predictions": rows}


def summary_lines(report: dict) -> List[str]:
    lines = []
    for key in [k for k in report if k.startswith("scenario:")] + ["scenario_average"]:
        r = report[key]
        lines.append(f"{key.split(':')[-1]:18s} Src {100 * r['source_acc']:5.1f}  "
                     f"Path {100 * r['pathology_acc']:5.1f}  Onset {100 * r['onset_acc']:5.1f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="trace JSONL or rl_*.parquet")
    ap.add_argument("--out", required=True, help="output prefix: <out>_report.json / <out>_predictions.jsonl")
    ap.add_argument("--backend", choices=("vllm", "hf", "api"), default="vllm")
    ap.add_argument("--model", default=None, help="checkpoint directory or Hub id, or the API model name")
    ap.add_argument("--lora", default=None, help="hf backend: PEFT adapter directory")
    ap.add_argument("--api-concurrency", type=int, default=4)
    ap.add_argument("--predictions", default=None, help="score this JSONL instead of generating")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="answer budget, default 128")
    ap.add_argument("--max-model-len", type=int, default=32768, help="vllm context, prompt plus answer")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing report")
    args = ap.parse_args()

    report_path, pred_path = f"{args.out}_report.json", f"{args.out}_predictions.jsonl"
    if os.path.exists(report_path) and not args.force:
        raise SystemExit(f"{report_path} exists; choose another --out or pass --force")
    items = load_items(args.data)[:args.limit]
    meta = {"data": os.path.basename(args.data), "n": len(items), "backend": args.backend}

    if args.predictions:
        by_id = {p["sample_id"]: p["output"] for p in read_jsonl(args.predictions)}
        items = [it for it in items if it["sample_id"] in by_id]
        outputs = [by_id[it["sample_id"]] for it in items]
        meta.update(backend=None, predictions=os.path.basename(args.predictions), n=len(items))
    else:
        if not args.model:
            raise SystemExit("--model is required")
        max_new_tokens = args.max_new_tokens or 128
        if args.backend == "api":
            from .api_client import APIClient
            client = APIClient(args.model, max_tokens=max_new_tokens)
            outputs = generate_api(client, items, args.api_concurrency, f"{args.out}_raw.jsonl")
        elif args.backend == "vllm":
            outputs = generate_vllm(args.model, items, max_new_tokens, args.tp, args.max_model_len)
        else:
            outputs = generate_hf(args.model, items, max_new_tokens, args.lora, args.batch_size)
        meta.update(model=args.model, lora=args.lora, max_new_tokens=max_new_tokens)

    result = score_predictions(items, outputs)
    result["report"]["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(pred_path, result["predictions"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result["report"], f, ensure_ascii=False, indent=2)
    print("\n".join(summary_lines(result["report"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
