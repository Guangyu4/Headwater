#!/bin/bash
# SFT of the auditor with verl's FSDP SFT trainer: full fine-tuning, lr 1e-5, 3 epochs,
# global batch 32, and verl's cosine schedule with 10% warmup, weight decay 0.01, and
# gradient clipping at 1.0. One checkpoint is saved per epoch so that the epoch can be
# chosen on the validation split. Micro batch 1 with sequence parallel 4 fits 32k-token
# traces on 80 GB GPUs, and truncation=error refuses to cut off an answer.
#
# Usage:  BASE_MODEL=Qwen/Qwen3-8B DATA_DIR=<headwater.to_parquet output> bash scripts/run_sft.sh <n_gpus> <save_dir>
set -euo pipefail
NGPUS="$1"
SAVE="$2"
: "${BASE_MODEL:?set BASE_MODEL}" "${DATA_DIR:?set DATA_DIR}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
TRAIN_BSZ="${TRAIN_BSZ:-32}"
SP_SIZE="${SP_SIZE:-4}"
SAVE_FREQ=$(python3 -c "import sys, pyarrow.parquet as pq; \
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows // int(sys.argv[2]))" "$DATA_DIR/sft_train.parquet" "$TRAIN_BSZ")

torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/sft_train.parquet" \
    data.val_files="$DATA_DIR/sft_val.parquet" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size="$TRAIN_BSZ" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    model.partial_pretrain="$BASE_MODEL" \
    model.trust_remote_code=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp \
    optim.lr="$LR" \
    ulysses_sequence_parallel_size="$SP_SIZE" \
    use_remove_padding=True \
    trainer.default_local_dir="$SAVE" \
    trainer.project_name=headwater \
    trainer.experiment_name="$(basename "$SAVE")" \
    trainer.logger=[console] \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$NGPUS" "${@:3}"
