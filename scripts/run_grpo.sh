#!/bin/bash
# GRPO of the auditor with verl and the field-wise reward of headwater/reward.py: LoRA 64/16,
# lr 5e-6, groups of 8 completions of at most 128 tokens, 128 prompts per step updated in
# mini-batches of 32, and a low-variance KL loss with coefficient 0.001. A checkpoint with a
# validation readout is saved every SAVE_TEST_FREQ steps so the step can be chosen on
# validation. The paper trains 6 epochs on Qwen3-8B and Llama-3.1-8B and 4 on Qwen3-14B.
# LoRA checkpoints become plain models with headwater.merge_lora.
#
# Usage:  INIT_MODEL=Qwen/Qwen3-8B DATA_DIR=<headwater.to_parquet output> bash scripts/run_grpo.sh <n_gpus> <save_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NGPUS="$1"
SAVE="$2"
: "${INIT_MODEL:?set INIT_MODEL}" "${DATA_DIR:?set DATA_DIR}"
MAX_PROMPT="${MAX_PROMPT:-32768}"
MAX_RESP="${MAX_RESP:-128}"
N_ROLLOUT="${N_ROLLOUT:-8}"
TRAIN_BSZ="${TRAIN_BSZ:-128}"
MINI_BSZ="${MINI_BSZ:-32}"
LORA_RANK="${LORA_RANK:-64}"
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-6}"
KL_COEF="${KL_COEF:-0.001}"
SAVE_TEST_FREQ="${SAVE_TEST_FREQ:-2}"
SEED="${SEED:-1}"
SP_SIZE="${SP_SIZE:-4}"
GPU_UTIL="${GPU_UTIL:-0.6}"
# one full-length sequence per micro-batch of each sequence-parallel group
TOKENS_PER_GPU=$(( (MAX_PROMPT + MAX_RESP + SP_SIZE - 1) / SP_SIZE ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/rl_train.parquet" \
    data.val_files="$DATA_DIR/rl_val.parquet" \
    data.prompt_key=prompt \
    data.train_batch_size="$TRAIN_BSZ" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESP" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    +data.seed="$SEED" \
    custom_reward_function.path="$ROOT/headwater/reward.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$INIT_MODEL" \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BSZ" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_UTIL" \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( MAX_PROMPT + MAX_RESP + 512 )) \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n="$N_ROLLOUT" \
    +actor_rollout_ref.rollout.seed="$SEED" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=[console] \
    trainer.project_name=headwater \
    trainer.experiment_name="$(basename "$SAVE")" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq="$SAVE_TEST_FREQ" \
    trainer.test_freq="$SAVE_TEST_FREQ" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir="$SAVE" \
    trainer.resume_mode=auto \
    trainer.total_epochs="$EPOCHS" "${@:3}"
