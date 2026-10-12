#!/bin/bash
# PMPO from an SFT checkpoint: read the SFT model's pathology confusions on validation, mine
# the three decoys of every training trace, and train with margin mu = 1, beta 0.1, LoRA
# rank 64, lr 5e-6, and 32 pairs per step for one epoch. The paper selects LoRA alpha on
# validation and uses 16 for Qwen3-8B and Llama-3.1-8B and 64 with --liger for Qwen3-14B.
#
# Usage:  SFT_MODEL=<sft checkpoint> TRACE_DIR=<dir with train.jsonl and val.jsonl> \
#             bash scripts/run_pmpo.sh <n_gpus> <out_dir> [extra run_pmpo arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
NGPUS="$1"
OUT="$2"
: "${SFT_MODEL:?set SFT_MODEL}" "${TRACE_DIR:?set TRACE_DIR}"
LORA_ALPHA="${LORA_ALPHA:-16}"
mkdir -p "$OUT"

[ -f "$OUT/sft_val_report.json" ] || \
    python -m headwater.evaluate --data "$TRACE_DIR/val.jsonl" --model "$SFT_MODEL" --out "$OUT/sft_val"
[ -f "$OUT/pairs.parquet" ] || \
    python -m headwater.mine_pairs --traces "$TRACE_DIR/train.jsonl" \
        --confusion-report "$OUT/sft_val_report.json" --out "$OUT/pairs.parquet"
accelerate launch --num_processes "$NGPUS" -m headwater.run_pmpo \
    --model "$SFT_MODEL" --pairs "$OUT/pairs.parquet" --out "$OUT/ckpt" \
    --margin-mu 1 --lora-alpha "$LORA_ALPHA" --grad-accum $(( 32 / NGPUS )) "${@:3}"
