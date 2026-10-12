"""Trace JSONL splits -> the two parquet layouts that verl reads.

  sft_<split>.parquet   extra_info.question is the prompt and extra_info.answer the label
                        JSON, so SFT learns the answer without a rationale
  rl_<split>.parquet    prompt is the chat message list, reward_model.ground_truth the label
                        JSON, and extra_info carries what headwater/reward.py needs

The input directory holds train.jsonl, val.jsonl, and test.jsonl in the format of
data/headwater_bench_sample.jsonl.

Run:  python -m headwater.to_parquet --trace-dir <dir with train/val/test.jsonl> --out-dir data/parquet
"""

import argparse
import json
import os

import pandas as pd

from .common import DATA_SOURCE, answer_json, build_prompt, read_jsonl, user_message

SPLITS = ("train", "val", "test")


def sft_row(sample: dict) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "extra_info": {
            "question": build_prompt(sample["input"]),
            "answer": answer_json(sample["answer"]),
            "sample_id": sample["sample_id"],
            "scenario": sample["scenario"],
        },
    }


def rl_row(sample: dict, index: int, split: str) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": user_message(build_prompt(sample["input"])),
        "ability": "audit",
        "reward_model": {"style": "rule", "ground_truth": answer_json(sample["answer"])},
        "extra_info": {
            "index": index,
            "split": split,
            "sample_id": sample["sample_id"],
            "scenario": sample["scenario"],
            "rounds": int(sample["rounds"]),
            "agents": list(sample["agents"]),
            # a JSON string because parquet structs need a fixed key set
            "swept_along_json": json.dumps(sample.get("swept_along") or {}),
            "network": sample.get("network") or "",
            "surface": sample.get("surface") or "",
            "position": sample.get("position") or "",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split in SPLITS:
        path = os.path.join(args.trace_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        samples = list(read_jsonl(path))
        pd.DataFrame([sft_row(s) for s in samples]).to_parquet(
            os.path.join(args.out_dir, f"sft_{split}.parquet"), index=False)
        pd.DataFrame([rl_row(s, i, split) for i, s in enumerate(samples)]).to_parquet(
            os.path.join(args.out_dir, f"rl_{split}.parquet"), index=False)
        print(f"{split}: {len(samples)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
