"""Merge a LoRA adapter into its base model and save a plain model directory.

verl writes a PEFT adapter under <ckpt>/global_step_N/actor/lora_adapter/ for LoRA runs, so
a GRPO checkpoint becomes a model that the vllm backend of headwater.evaluate can load.

Run:  python -m headwater.merge_lora --actor <ckpt>/global_step_N/actor --out <ckpt>/merged_step_N
"""

import argparse
import json
import os
import shutil


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", default=None, help="verl actor directory with lora_adapter/")
    ap.add_argument("--adapter", default=None, help="a PEFT adapter directory instead of --actor")
    ap.add_argument("--out", required=True, help="target model directory, which must not exist")
    ap.add_argument("--base", default=None, help="base model, by default the adapter's base_model_name_or_path")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = ap.parse_args()

    if bool(args.actor) == bool(args.adapter):
        raise SystemExit("pass exactly one of --actor and --adapter")
    adapter = args.adapter or os.path.join(args.actor, "lora_adapter")
    if not os.path.isdir(adapter):
        raise SystemExit(f"no adapter at {adapter}")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    with open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8") as f:
        base = args.base or json.load(f).get("base_model_name_or_path")
    if not base:
        raise SystemExit("base model unknown; pass --base")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    tmp = args.out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    src = args.actor or args.adapter
    tokenizer_dir = src if os.path.exists(os.path.join(src, "tokenizer_config.json")) else base
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(tmp)
    os.rename(tmp, args.out)
    print(f"merged model written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
