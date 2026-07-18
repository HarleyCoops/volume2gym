"""Portable exports for supervised and reinforcement-learning trainers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from volume2gym.models import StructuredAnswer, Task

DEFAULT_SYSTEM_PROMPT = (
    "Use the supplied structured volume. Return a source-grounded answer that follows "
    "the requested JSON contract."
)


def reference_answer(task: Task) -> StructuredAnswer:
    key = task.answer_key
    final = key.reference_answer or "; ".join(key.required_actions)
    return StructuredAnswer(
        applicable_rules=key.applicable_unit_ids,
        situation_type=task.task_family.value,
        required_actions=key.required_actions,
        forbidden_actions=key.forbidden_actions,
        procedure_order=key.procedure_order,
        final_answer=final,
    )


def sft_record(task: Task, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, Any]:
    answer = reference_answer(task).model_dump(mode="json")
    return {
        "id": task.task_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
        ],
        "metadata": {
            "knowledge_unit_ids": list(task.knowledge_unit_ids),
            "task_family": task.task_family.value,
            "split": task.split.value if task.split else None,
        },
    }


def grpo_record(task: Task) -> dict[str, Any]:
    """Return a prompt plus private verifier contract for GRPO-style trainers."""

    return {
        "id": task.task_id,
        "prompt": task.prompt,
        "answer_contract": task.answer_contract.model_dump(mode="json"),
        "answer_key": task.answer_key.model_dump(mode="json"),
        "metadata": {
            "knowledge_unit_ids": list(task.knowledge_unit_ids),
            "task_family": task.task_family.value,
            "split": task.split.value if task.split else None,
        },
    }


def write_sft_jsonl(tasks: Iterable[Task], path: str | Path) -> Path:
    return _write_records((sft_record(task) for task in tasks), path)


def write_grpo_jsonl(tasks: Iterable[Task], path: str | Path) -> Path:
    return _write_records((grpo_record(task) for task in tasks), path)


def _write_records(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "grpo_record",
    "reference_answer",
    "sft_record",
    "write_grpo_jsonl",
    "write_sft_jsonl",
]
