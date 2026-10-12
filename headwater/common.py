"""The auditor prompt, the canonical answer JSON, and JSONL helpers.

One prompt builder serves SFT, GRPO, and PMPO alike, so the three paradigms see the
same inputs and differ only in the learning signal. The instruction names the three
fields, the six pathologies, and the answer format, and lists exactly the blocks that
the record carries. It never says how faults are produced.

Evaluation states that exactly one participant deviated and drops the all-null option
(`assume_fault`), so every auditor is scored on attribution alone.
"""

import json
from typing import Iterable, Iterator, List

from .reward import KEYS, PATHOLOGIES

DATA_SOURCE = "headwater/social_sim"

_TASK_TAIL = (
    "At most ONE participant may have started behaving abnormally from some round "
    "onward, in one of these ways: " + ", ".join(PATHOLOGIES) + ". "
    "It is also possible that nobody did.\n\n"
    "Answer with a single JSON object and nothing else, exactly in this form:\n"
    '{"source": "<agent id>", "pathology": "<one of the six names>", "onset": <round index>}\n'
    "where onset is the first round (as numbered in the record, e.g. r3 -> 3) in "
    "which the source's behavior changed. If nobody behaved abnormally, answer\n"
    '{"source": null, "pathology": null, "onset": null}'
)

_ROLE = "You are auditing a round-based social simulation run by language-model agents. "

# Record blocks in the order a record lists them, each with the words the instruction uses for it.
_BLOCKS = (
    ("=== PERSONA CARDS (baseline) ===", ("the participants' persona cards as configured before the run",)),
    ("=== NETWORK EDGES ===", ("the communication network (who reads whom)",)),
    ("=== TRANSCRIPT ===", ("the full transcript by round",)),
    ("=== DECLARED VALUES BY ROUND ===", ("a table of every participant's declared value in every round",
                                         "excerpts of the transcript (a subset of the turns; the table covers the rest)")),
    ("=== MACRO CURVES ===", ("the per-round macro curves",)),
)

_ASSUME_FAULT_EDITS = (
    ("At most ONE participant may have started behaving abnormally from some round onward",
     "Exactly ONE participant started behaving abnormally from some round onward"),
    (" It is also possible that nobody did.", ""),
    (' If nobody behaved abnormally, answer\n{"source": null, "pathology": null, "onset": null}', ""),
)


def instruction_for(input_text: str) -> str:
    """The task instruction naming exactly the blocks that the record carries."""
    items = [phrase for marker, phrases in _BLOCKS if marker in input_text for phrase in phrases]
    if not items:
        raise ValueError("record carries none of the known blocks")
    listing = items[0] if len(items) == 1 else ", ".join(items[:-1]) + ", and " + items[-1]
    return f"{_ROLE}Below are {listing}.\n\n{_TASK_TAIL}"


def build_prompt(input_text: str) -> str:
    return f"{instruction_for(input_text)}\n\n{input_text.strip()}\n"


def assume_fault(prompt: str) -> str:
    """The evaluation prompt: the instruction states that exactly one participant deviated
    and offers no all-null answer. The record itself is untouched."""
    for old, new in _ASSUME_FAULT_EDITS:
        if prompt.count(old) != 1:
            raise ValueError(f"instruction sentence not found exactly once: {old[:50]!r}")
        prompt = prompt.replace(old, new)
    return prompt


def answer_json(answer: dict) -> str:
    """Compact JSON with a fixed key order, the SFT target and the PMPO chosen answer."""
    return json.dumps({k: answer.get(k) for k in KEYS}, ensure_ascii=False)


def user_message(prompt: str) -> List[dict]:
    return [{"role": "user", "content": prompt}]


def render_chat(tokenizer, messages: List[dict]) -> str:
    """Chat template with the generation prompt on and thinking off, as verl applies it,
    so SFT, GRPO, PMPO, and evaluation all see the same prompt."""
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
