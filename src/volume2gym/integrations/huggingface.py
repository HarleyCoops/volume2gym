"""Import the published Hugging Face contract fixture into canonical artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from volume2gym.models import (
    AnswerKey,
    Citation,
    KnowledgeKind,
    KnowledgeUnit,
    Split,
    Task,
    TaskFamily,
)
from volume2gym.sources import read_json, read_jsonl


@dataclass(frozen=True, slots=True)
class HuggingFaceFixture:
    knowledge_units: tuple[KnowledgeUnit, ...]
    tasks: tuple[Task, ...]
    adapter_config: dict[str, Any]
    adapter_weights: dict[str, Any]


def load_huggingface_fixture(directory: str | Path) -> HuggingFaceFixture:
    """Load every task row without assuming a particular rule number."""

    root = Path(directory)
    config = read_json(root / "adapter_config.json")
    weights = read_json(root / "adapter_weights.json")
    train_rows = list(read_jsonl(root / "training_dataset.jsonl"))
    heldout_rows = list(read_jsonl(root / "heldout_eval.jsonl"))
    rows = [(item, Split.TRAIN) for item in train_rows] + [
        (item, Split.TEST) for item in heldout_rows
    ]

    unit_records: dict[str, dict[str, Any]] = {}
    for row, _split in rows:
        for rule_id in row.get("expected_rule_ids", ()):
            unit_id = canonical_unit_id(rule_id)
            unit_records.setdefault(
                unit_id,
                {
                    "rule_id": str(rule_id),
                    "row": row,
                },
            )

    units = tuple(
        _unit_from_fixture(unit_id, record["rule_id"], record["row"], weights)
        for unit_id, record in sorted(unit_records.items())
    )
    tasks = tuple(
        _task_from_row(row, split=split)
        for row, split in rows
    )
    return HuggingFaceFixture(
        knowledge_units=units,
        tasks=tasks,
        adapter_config=config,
        adapter_weights=weights,
    )


def canonical_unit_id(identifier: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(identifier).casefold()).strip("-")
    if not slug:
        raise ValueError("knowledge-unit identifier cannot be empty")
    return f"source-unit-{slug}"


def _unit_from_fixture(
    unit_id: str,
    rule_id: str,
    row: dict[str, Any],
    weights: dict[str, Any],
) -> KnowledgeUnit:
    citations = tuple(_citation(item) for item in row.get("source_citations", ()))
    if not citations:
        raise ValueError(f"fixture unit {unit_id} has no source citation")
    matching_weights = weights if canonical_unit_id(weights.get("rule_id", "")) == unit_id else {}
    return KnowledgeUnit(
        unit_id=unit_id,
        kind=KnowledgeKind.RULE,
        title=rule_id,
        text="Source text is referenced by provenance but is not redistributed in this fixture.",
        family=row.get("rule_family"),
        section=citations[0].section,
        citations=citations,
        conditions=tuple((matching_weights.get("condition_weights") or {}).keys()),
        required_actions=tuple(row.get("expected_required_actions", ())),
        forbidden_actions=tuple(row.get("expected_forbidden_actions", ())),
        procedure_steps=tuple(row.get("expected_procedure_order", ())),
        terms=tuple(row.get("expected_terms", ())),
        confidence=min(citation.confidence for citation in citations),
        metadata={"upstream_rule_id": rule_id, "fixture_only": True},
    )


def _task_from_row(row: dict[str, Any], *, split: Split) -> Task:
    citations = tuple(_citation(item) for item in row.get("source_citations", ()))
    unit_ids = tuple(canonical_unit_id(item) for item in row.get("expected_rule_ids", ()))
    if not unit_ids:
        raise ValueError(f"fixture task {row.get('task_id', '<unknown>')} has no expected rule")
    try:
        family = TaskFamily(str(row.get("task_type")))
    except ValueError:
        family = TaskFamily.APPLIED_SCENARIO
    required = tuple(row.get("expected_required_actions", ()))
    reference = "; ".join(required)
    return Task(
        task_id=str(row["task_id"]),
        prompt=str(row["prompt"]),
        task_family=family,
        knowledge_unit_ids=unit_ids,
        answer_key=AnswerKey(
            applicable_unit_ids=unit_ids,
            required_actions=required,
            forbidden_actions=tuple(row.get("expected_forbidden_actions", ())),
            procedure_order=tuple(row.get("expected_procedure_order", ())),
            terms=tuple(row.get("expected_terms", ())),
            reference_answer=reference or None,
            citations=citations,
        ),
        citations=citations,
        rule_family=row.get("rule_family"),
        tags=("huggingface-fixture", "one-rule-contract"),
        generator="huggingface:volume2gym-railroad-1959",
        split=split,
        metadata={"upstream_task_id": row["task_id"]},
    )


def _citation(record: dict[str, Any]) -> Citation:
    title = str(record.get("volume_title") or "source-volume")
    document_id = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    return Citation(
        document_id=document_id,
        span_id=str(record.get("source_span") or record.get("rule_id") or "unknown-span"),
        page_number=record.get("page_number"),
        section=record.get("section"),
        locator=record.get("rule_id"),
        confidence=float(record.get("confidence", 1.0)),
    )


__all__ = ["HuggingFaceFixture", "canonical_unit_id", "load_huggingface_fixture"]
