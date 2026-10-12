# What's in the Water: Auditing Faults in LLM Social Simulations

This anonymous repository accompanies the paper under review. It holds the evaluation and
training code of the auditors, a sample of Headwater-Bench and of the out-of-distribution
test set, and answers of every auditor in the paper on part of these traces. The full
benchmark follows the review period.

## Contents

| Path | Content |
|---|---|
| `headwater/evaluate.py` | evaluation through an API, vLLM, or transformers, and scoring of existing answers |
| `headwater/api_client.py` | OpenAI-compatible client with retries |
| `headwater/common.py` | auditor prompt and data helpers |
| `headwater/reward.py` | answer parsing, format gate, and field-wise reward |
| `headwater/heuristics.py` | rule-based baselines |
| `headwater/to_parquet.py` | trace splits to the SFT and RL parquet files of verl |
| `headwater/mine_pairs.py` | PMPO preference pairs mined from the labels |
| `headwater/run_pmpo.py` | PMPO trainer on TRL |
| `headwater/merge_lora.py` | merge a LoRA checkpoint into its base model |
| `scripts/run_sft.sh`, `scripts/run_grpo.sh`, `scripts/run_pmpo.sh` | training launchers |
| `data/headwater_bench_sample.jsonl` | 96 faulty test traces of Headwater-Bench, 24 per scenario |
| `data/ood_sample.jsonl` | 48 faulty traces of the two OOD scenarios, 24 per scenario |
| `responses/headwater_bench/<model>.jsonl` | answers of one auditor on 20 random traces of the Headwater-Bench sample |
| `responses/ood/<model>.jsonl` | answers of one auditor on 10 random traces of the OOD sample |

```bash
pip install -r requirements.txt
```

## Task and Metrics

Each trace comes from a round-based LLM social simulation in which exactly one agent was
made to deviate from some round onward. An auditor reads the serialized trace and names the
**source** agent, the **pathology**, and the **onset** round, answering

```json
{"source": "<agent id>", "pathology": "<one of the six names>", "onset": <round index>}
```

Source, Pathology, and Onset are exact-match accuracies over faulty traces. An answer that
is not such an object, or names an unknown agent, an unknown pathology, or a round outside
the trace, scores zero on every field. The scenario average weights the scenarios equally.

## Evaluation

```bash
# an API model behind any OpenAI-compatible endpoint
export OPENAI_BASE_URL=<endpoint> OPENAI_API_KEY=<key>
python -m headwater.evaluate --data data/headwater_bench_sample.jsonl --backend api --model <model name> --out eval/api

# a local checkpoint with vLLM
python -m headwater.evaluate --data data/headwater_bench_sample.jsonl --model Qwen/Qwen3-8B --out eval/qwen3-8b

# existing answers, for example the released ones
python -m headwater.evaluate --data data/headwater_bench_sample.jsonl \
    --predictions responses/headwater_bench/headwater-8b.jsonl --out eval/headwater-8b

# a rule-based baseline
python -m headwater.heuristics --data data/ood_sample.jsonl --strategy deviant --out eval/deviant.jsonl
```

Each run writes `<out>_report.json` with the accuracies per scenario, their average, the
slices by pathology, network, surface, source position, and onset tier, and the pathology
confusion matrix, together with `<out>_predictions.jsonl`. An API model whose first answer
fails the format check is asked once more with the format restated, and API answers are
cached in `<out>_raw.jsonl`, so an interrupted run resumes where it stopped.

## Training

Training expects a directory with `train.jsonl`, `val.jsonl`, and `test.jsonl` in the trace
format below. The launchers default to the hyper-parameters of the paper.

```bash
python -m headwater.to_parquet --trace-dir <trace dir> --out-dir data/parquet

# SFT with verl, full fine-tuning
BASE_MODEL=Qwen/Qwen3-8B DATA_DIR=data/parquet bash scripts/run_sft.sh 4 runs/sft

# GRPO with verl and the field-wise reward, LoRA checkpoints merged before evaluation
INIT_MODEL=Qwen/Qwen3-8B DATA_DIR=data/parquet bash scripts/run_grpo.sh 4 runs/grpo
python -m headwater.merge_lora --actor runs/grpo/global_step_42/actor --out runs/grpo/merged_step_42

# PMPO from the SFT checkpoint chosen on validation
SFT_MODEL=runs/sft/global_step_93 TRACE_DIR=<trace dir> bash scripts/run_pmpo.sh 4 runs/pmpo
```

`run_pmpo.sh` evaluates the SFT model on the validation split, mines a source, an onset, and
a pathology decoy for every training trace, and trains with the reward gap of each decoy as
its margin.

## Trace Format

One JSON object per line.

| Field | Meaning |
|---|---|
| `sample_id` | opaque id, `<scenario>-<index>` |
| `scenario` | `opinion`, `diffusion`, `pgg` for Public Goods, `deliberation`, and for OOD `recfeed` for recommendation feeds and `bargain` for bargaining |
| `network` | `erdos_renyi`, `planted_hub`, `two_community`, `complete`, or `committee` |
| `position` | network position of the source, `hub`, `bridge`, or `periphery` |
| `surface` | where the pathology takes effect in the source, `perception`, `persona`, `memory`, or `response` |
| `rounds` | number of rounds, numbered from 0 |
| `agents` | agent ids |
| `edges` | undirected communication edges |
| `swept_along` | the source with its onset and every peer the fault swept along with the round it was reached |
| `input` | the serialized trace the auditor reads, with persona cards, network edges, declared values by round, transcript excerpts, and macro curves |
| `answer` | the label `{"source", "pathology", "onset"}` |

The six pathologies are `conformity`, `sycophancy`, `persona_drift`,
`memory_inconsistency`, `relay_distortion`, and `stereotyping`.

## Released Answers

One JSON object `{"sample_id", "output"}` per line, where `output` is the verbatim answer
that was scored and `sample_id` refers to a trace in the matching data file.

| File | Auditor |
|---|---|
| `gpt-5.6-sol` | GPT-5.6 Sol |
| `gpt-5.5` | GPT-5.5 |
| `claude-sonnet-5` | Claude-Sonnet-5 |
| `claude-sonnet-4.6` | Claude-Sonnet-4.6 |
| `deepseek-v4-pro` | DeepSeek-V4-Pro |
| `qwen3-8b`, `qwen3-8b-sft`, `qwen3-8b-grpo` | Qwen3-8B zero-shot, with SFT, and with GRPO |
| `headwater-qwen-8b` | Headwater-Qwen-8B, Qwen3-8B with PMPO |
| `qwen3-14b`, `qwen3-14b-sft`, `qwen3-14b-grpo` | Qwen3-14B zero-shot, with SFT, and with GRPO |
| `headwater-qwen-14b` | Headwater-Qwen-14B, Qwen3-14B with PMPO |
| `llama-3.1-8b`, `llama-3.1-8b-sft`, `llama-3.1-8b-grpo` | Llama-3.1-8B-Instruct zero-shot, with SFT, and with GRPO |
| `headwater-8b` | Headwater-8B, Llama-3.1-8B-Instruct with PMPO |
