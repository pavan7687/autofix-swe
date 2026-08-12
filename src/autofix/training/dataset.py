"""Turning JSONL examples into tokenized, loss-masked training tensors.

The single most important detail here is **completion-only loss masking**. If
loss is computed over the prompt as well as the answer, the model spends most
of its capacity learning to reproduce bug reports and source code — text it
will always be *given* at inference, never asked to produce. Masking the prompt
to -100 concentrates the gradient on the patch, which is the only thing we
actually want it to learn.

The second is that prompts are built by `autofix.prompting`, never inline, so
training and inference cannot drift apart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autofix.logging_conf import get_logger
from autofix.models import EditingExample, RetrievalExample
from autofix.prompting import edit_messages, rerank_messages

log = get_logger(__name__)

IGNORE_INDEX = -100


def read_jsonl(path: Path, model_cls: type) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(model_cls.model_validate_json(line))
    log.info("dataset.read", path=str(path), rows=len(rows))
    return rows


@dataclass
class MaskedSFTDataset:
    """Chat-formatted examples with loss masked to the assistant turn only."""

    examples: list
    tokenizer: Any
    max_seq_len: int
    task: str  # "editing" | "retrieval"
    _skipped: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    def _messages(self, example) -> list[dict[str, str]]:
        return edit_messages(example) if self.task == "editing" else rerank_messages(example)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        messages = self._messages(self.examples[idx])
        prompt_messages, answer = messages[:-1], messages[-1]

        # Render the prompt exactly as generation would, then the full
        # conversation. The difference in length is the answer span.
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + answer["content"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            full_text, add_special_tokens=False, truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def stats(self) -> dict[str, int]:
        lengths = []
        for i in range(min(len(self), 500)):
            lengths.append(len(self[i]["input_ids"]))
        lengths.sort()
        if not lengths:
            return {}
        return {
            "sampled": len(lengths),
            "p50_tokens": lengths[len(lengths) // 2],
            "p90_tokens": lengths[int(len(lengths) * 0.9)],
            "max_tokens": lengths[-1],
            "at_truncation_limit": sum(1 for x in lengths if x >= self.max_seq_len),
        }


@dataclass
class PadCollator:
    """Right-pads a batch and pads labels with IGNORE_INDEX, not the pad token.

    Padding labels with the tokenizer's pad id would train the model to emit
    padding; -100 tells the loss to skip those positions entirely.
    """

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def load_task_dataset(
    data_dir: Path, split: str, task: str, tokenizer: Any, max_seq_len: int
) -> MaskedSFTDataset:
    model_cls = EditingExample if task == "editing" else RetrievalExample
    examples = read_jsonl(data_dir / f"{split}.jsonl", model_cls)
    return MaskedSFTDataset(examples, tokenizer, max_seq_len, task)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
