"""PMPO: preference optimization on pollution-mined pairs with TRL's DPOTrainer.

The policy and the frozen reference both start from the SFT checkpoint, since DPO needs a
reference that already emits the answer format. Pairs are converted to TRL's standard
format with the chat template applied by common.render_chat, so the prompt matches the one
SFT and GRPO saw.

Loss: -log sigma(beta * delta - m) with the per-pair target margin m = mu * reward_gap from
headwater.mine_pairs, and --margin-mu 0 recovers sigmoid DPO. TRL has no per-pair margin,
so MarginDPOTrainer carries a `margin` column through collation and applies it in
`dpo_loss`. Only the sigmoid loss is supported.

Run:  accelerate launch --num_processes 4 -m headwater.run_pmpo \
          --model <sft checkpoint> --pairs pairs.parquet --out runs/pmpo --margin-mu 1
"""

import argparse
import glob
import os


def to_standard(ds, tokenizer, margin_mu=0.0):
    """Conversational to standard format: the rendered prompt, the answers followed by EOS,
    and margin = mu * reward_gap."""
    from .common import render_chat
    eos = tokenizer.eos_token or ""

    def convert(row):
        return {
            "prompt": render_chat(tokenizer, list(row["prompt"])),
            "chosen": row["chosen"][0]["content"] + eos,
            "rejected": row["rejected"][0]["content"] + eos,
            "margin": float(margin_mu) * float(row["reward_gap"]),
        }
    return ds.map(convert, remove_columns=ds.column_names)


def margin_trainer_classes():
    """(MarginDPOTrainer, MarginCollator), built lazily so that this module imports without trl."""
    import torch
    import torch.nn.functional as F
    from trl import DPOTrainer
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    class MarginCollator(DataCollatorForPreference):
        def torch_call(self, examples):
            batch = super().torch_call(examples)
            batch["margin"] = torch.tensor([float(ex["margin"]) for ex in examples], dtype=torch.float32)
            return batch

    class MarginDPOTrainer(DPOTrainer):
        _batch_margin = None

        def _set_signature_columns_if_needed(self):
            # keep the margin column through remove_unused_columns so it reaches the collator
            super()._set_signature_columns_if_needed()
            if "margin" not in self._signature_columns:
                self._signature_columns.append("margin")

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            self._batch_margin = batch.get("margin")
            try:
                loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
            finally:
                self._batch_margin = None
            if "margin" in batch:
                prefix = "eval_" if train_eval == "eval" else ""
                metrics[f"{prefix}margin/mean"] = self.accelerator.gather_for_metrics(
                    batch["margin"].to(self.accelerator.device)).mean().item()
            return loss, metrics

        def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
            if self._batch_margin is None:
                return super().dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
            device = self.accelerator.device
            chosen_logratios = chosen_logps.to(device) - ref_chosen_logps.to(device)
            rejected_logratios = rejected_logps.to(device) - ref_rejected_logps.to(device)
            margin = self._batch_margin.to(device=device, dtype=chosen_logratios.dtype)
            losses = -F.logsigmoid(self.beta * (chosen_logratios - rejected_logratios) - margin)
            return losses, self.beta * chosen_logratios.detach(), self.beta * rejected_logratios.detach()

    return MarginDPOTrainer, MarginCollator


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="SFT checkpoint, the policy init and the reference")
    ap.add_argument("--pairs", required=True, help="parquet of headwater.mine_pairs")
    ap.add_argument("--eval-pairs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-mu", type=float, default=1.0, help="mu of the margin m = mu * reward_gap, 0 = DPO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8, help="1 x 8 x 4 GPUs = 32 pairs per step")
    ap.add_argument("--lora-rank", type=int, default=64, help="0 = full fine-tuning")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--precompute-ref", action="store_true", help="precompute reference log-probs once")
    ap.add_argument("--liger", action="store_true", help="fused Liger SwiGLU, RMSNorm, and RoPE kernels")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint-* in --out")
    args = ap.parse_args()

    existing = sorted(glob.glob(os.path.join(args.out, "checkpoint-*")))
    if existing and not args.resume:
        raise SystemExit(f"{args.out} already holds checkpoints; pass --resume or use a new --out")

    import functools

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig
    MarginDPOTrainer, MarginCollator = margin_trainer_classes()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_files = {"train": args.pairs}
    if args.eval_pairs:
        data_files["eval"] = args.eval_pairs
    ds = load_dataset("parquet", data_files=data_files)
    ds = {split: to_standard(d, tokenizer, args.margin_mu) for split, d in ds.items()}

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="flash_attention_2")
    if args.liger:
        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model, rope=True, swiglu=True, rms_norm=True,
                                        cross_entropy=False, fused_linear_cross_entropy=False)

    # The sigmoid loss needs the logits only, so the hidden states that TRL requests are
    # switched off. The forward is wrapped because PEFT calls it directly and skips hooks.
    base_forward = model.forward

    @functools.wraps(base_forward)
    def forward_without_hidden_states(*f_args, **f_kwargs):
        f_kwargs["output_hidden_states"] = False
        return base_forward(*f_args, **f_kwargs)

    model.forward = forward_without_hidden_states
    peft_config = None
    if args.lora_rank > 0:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    config = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        # traces are long: no truncation, since TRL would otherwise keep only the end of the prompt
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        use_logits_to_keep=True,
        precompute_ref_log_probs=args.precompute_ref,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if args.eval_pairs else "no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    if (config.loss_type not in ("sigmoid", ["sigmoid"]) or config.label_smoothing or config.reference_free
            or config.rpo_alpha is not None or config.use_weighting):
        raise SystemExit("MarginDPOTrainer implements the plain sigmoid loss only")
    trainer = MarginDPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tokenizer,
        data_collator=MarginCollator(pad_token_id=tokenizer.pad_token_id),
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=bool(existing) if args.resume else None)
    final = os.path.join(args.out, "final")
    if peft_config is not None:
        # final/ is a plain model directory with the adapter merged into the SFT weights
        trainer.save_model(os.path.join(args.out, "final_adapter"))
        if trainer.is_world_process_zero():
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(final, safe_serialization=True, max_shard_size="5GB")
            tokenizer.save_pretrained(final)
    else:
        trainer.save_model(final)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
