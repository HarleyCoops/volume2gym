"""Importer for the legacy Railroad Engineer 1959 JSON artifacts.

The adapter treats every rule identically. It contains no rule-number fixtures
or special cases; nested sub-rules are recursively normalized rather than lost.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..models import (
    AnswerContract,
    AnswerKey,
    Citation,
    KnowledgeKind,
    KnowledgeUnit,
    Task,
    TaskFamily,
)
from . import ImportedVolume

DEFAULT_DOCUMENT_ID = "railroad-1959"


class RailroadProfile:
    """Normalize the original railroad rule and task JSON schemas."""

    profile_id = "railroad-1959"

    def import_units(
        self,
        payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        document_id: str = DEFAULT_DOCUMENT_ID,
    ) -> list[KnowledgeUnit]:
        records = _unwrap_records(payload, "rules")
        units: list[KnowledgeUnit] = []
        seen: dict[str, str] = {}

        def visit(
            record: Mapping[str, Any],
            *,
            inherited_family: str | None = None,
            inherited_section: str | None = None,
            inherited_page_number: int | None = None,
            parent_unit_id: str | None = None,
        ) -> None:
            legacy_id = _required_text(record, "rule_id")
            unit_id = _unit_id(legacy_id)
            if unit_id in seen:
                raise ValueError(
                    f"duplicate or colliding railroad rule id {legacy_id!r}; "
                    f"already used by {seen[unit_id]!r}"
                )
            seen[unit_id] = legacy_id

            text = _required_text(record, "text")
            family = _optional_text(record.get("category")) or inherited_family
            section = _optional_text(record.get("section")) or inherited_section
            page_number = _optional_int(record.get("page_number")) or inherited_page_number
            confidence = float(record.get("confidence", 1.0))
            citation = Citation(
                document_id=document_id,
                span_id=f"{document_id}:rule:{legacy_id}",
                page_number=page_number,
                section=section,
                locator=f"Rule {legacy_id}",
                quote=text,
                confidence=confidence,
            )
            metadata: dict[str, Any] = {
                "legacy_rule_id": legacy_id,
                "source_format": "railroad_rule_json",
            }
            if parent_unit_id:
                metadata["parent_unit_id"] = parent_unit_id
            if page_number is not None:
                metadata["page_number"] = page_number

            unit = KnowledgeUnit(
                unit_id=unit_id,
                kind=KnowledgeKind("rule"),
                title=_optional_text(record.get("title")) or f"Rule {legacy_id}",
                text=text,
                family=family,
                section=section,
                citations=(citation,),
                conditions=_strings(record.get("conditions")),
                required_actions=_strings(record.get("required_actions")),
                forbidden_actions=_strings(record.get("forbidden_actions")),
                procedure_steps=_strings(record.get("procedure_steps") or record.get("procedure")),
                exceptions=_strings(record.get("exceptions")),
                terms=_strings(record.get("terms")),
                related_unit_ids=tuple(
                    _unit_id(item)
                    for item in _strings(
                        record.get("related_rule_ids") or record.get("related_unit_ids")
                    )
                ),
                confidence=confidence,
                metadata=metadata,
            )
            units.append(unit)

            children = record.get("sub_rules") or ()
            if isinstance(children, (str, bytes, Mapping)):
                raise ValueError(f"sub_rules for {legacy_id!r} must be a sequence of objects")
            for child in children:
                if not isinstance(child, Mapping):
                    raise ValueError(f"sub-rule of {legacy_id!r} must be an object")
                visit(
                    child,
                    inherited_family=family,
                    inherited_section=section,
                    inherited_page_number=page_number,
                    parent_unit_id=unit_id,
                )

        for record in records:
            visit(record)
        return units

    def import_rules(
        self,
        payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        document_id: str = DEFAULT_DOCUMENT_ID,
    ) -> list[KnowledgeUnit]:
        """Backward-readable alias for :meth:`import_units`."""

        return self.import_units(payload, document_id=document_id)

    def import_tasks(
        self,
        payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        units: Sequence[KnowledgeUnit] = (),
        document_id: str = DEFAULT_DOCUMENT_ID,
    ) -> list[Task]:
        records = _unwrap_records(payload, "tasks")
        unit_by_legacy_id = {
            str(unit.metadata.get("legacy_rule_id")): unit
            for unit in units
            if unit.metadata.get("legacy_rule_id") is not None
        }
        seen_task_ids: set[str] = set()
        tasks: list[Task] = []

        for record in records:
            task_id = _required_text(record, "task_id")
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate railroad task id {task_id!r}")
            seen_task_ids.add(task_id)

            legacy_rule_ids = _strings(record.get("applicable_rules"))
            if not legacy_rule_ids:
                raise ValueError(f"railroad task {task_id!r} has no applicable_rules")
            applicable_unit_ids = tuple(
                unit_by_legacy_id[legacy_id].unit_id
                if legacy_id in unit_by_legacy_id
                else _unit_id(legacy_id)
                for legacy_id in legacy_rule_ids
            )
            applicable_units = [
                unit_by_legacy_id[legacy_id]
                for legacy_id in legacy_rule_ids
                if legacy_id in unit_by_legacy_id
            ]
            citations = _unique_citations(
                citation
                for legacy_id in legacy_rule_ids
                for citation in (
                    unit_by_legacy_id[legacy_id].citations
                    if legacy_id in unit_by_legacy_id
                    else (
                        Citation(
                            document_id=document_id,
                            span_id=f"{document_id}:rule:{legacy_id}",
                            locator=f"Rule {legacy_id}",
                        ),
                    )
                )
            )
            families = sorted({unit.family for unit in applicable_units if unit.family})
            rule_family = "+".join(families) if families else None
            task_family = _task_family(record)

            answer_key = AnswerKey(
                applicable_unit_ids=applicable_unit_ids,
                required_actions=_strings(record.get("required_actions")),
                forbidden_actions=_strings(record.get("forbidden_actions")),
                procedure_order=_strings(record.get("procedure_order") or record.get("procedure")),
                terms=_strings(record.get("terms")),
                reference_answer=_required_text(record, "expected_outcome"),
                citations=citations,
            )
            tags = tuple(
                dict.fromkeys(("railroad", "legacy-import", *_strings(record.get("tags"))))
            )
            tasks.append(
                Task(
                    task_id=task_id,
                    prompt=_required_text(record, "description"),
                    task_family=task_family,
                    knowledge_unit_ids=applicable_unit_ids,
                    answer_contract=AnswerContract(),
                    answer_key=answer_key,
                    citations=citations,
                    rule_family=rule_family,
                    tags=tags,
                    generator="legacy:railroad_extraction",
                    metadata={
                        "legacy_task_id": task_id,
                        "source_format": "railroad_safety_task_json",
                        "document_id": document_id,
                    },
                )
            )
        return tasks

    def import_bundle(
        self,
        rules: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        *,
        document_id: str = DEFAULT_DOCUMENT_ID,
    ) -> ImportedVolume:
        units = self.import_units(rules, document_id=document_id)
        imported_tasks = (
            self.import_tasks(tasks, units=units, document_id=document_id)
            if tasks is not None
            else []
        )
        return ImportedVolume(tuple(units), tuple(imported_tasks))


Railroad1959Profile = RailroadProfile


def import_legacy_rules(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    document_id: str = DEFAULT_DOCUMENT_ID,
) -> list[KnowledgeUnit]:
    return RailroadProfile().import_units(payload, document_id=document_id)


def import_legacy_tasks(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    units: Sequence[KnowledgeUnit] = (),
    document_id: str = DEFAULT_DOCUMENT_ID,
) -> list[Task]:
    return RailroadProfile().import_tasks(payload, units=units, document_id=document_id)


def _unwrap_records(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    wrapper_key: str,
) -> list[Mapping[str, Any]]:
    raw: Any = payload.get(wrapper_key, [payload]) if isinstance(payload, Mapping) else payload
    if isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"{wrapper_key} payload must be a sequence of objects")
    records = list(raw)
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError(f"{wrapper_key} payload must contain only objects")
    return records


def _unit_id(legacy_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(legacy_id).strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"railroad rule id {legacy_id!r} cannot be normalized")
    return f"railroad-rule-{slug}"


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = _optional_text(record.get(key))
    if value is None:
        raise ValueError(f"legacy railroad record is missing non-empty {key!r}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        raise ValueError(f"expected text or sequence of text, got {type(value).__name__}")
    return tuple(text for item in value if (text := _optional_text(item)) is not None)


def _task_family(record: Mapping[str, Any]) -> TaskFamily:
    candidate = _optional_text(record.get("task_family") or record.get("task_type"))
    if candidate:
        normalized = candidate.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "standard": "standard_operation",
            "edge": "edge_case",
            "violation": "violation_check",
            "conflict": "conflict_resolution",
            "exception": "exception_handling",
            "adversarial": "adversarial_distractor",
        }
        try:
            return TaskFamily(aliases.get(normalized, normalized))
        except ValueError:
            pass
    return TaskFamily("applied_scenario")


def _unique_citations(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    result: list[Citation] = []
    seen: set[tuple[str, str, str | None]] = set()
    for citation in citations:
        key = (citation.document_id, citation.span_id, citation.locator)
        if key not in seen:
            seen.add(key)
            result.append(citation)
    return tuple(result)


__all__ = [
    "DEFAULT_DOCUMENT_ID",
    "Railroad1959Profile",
    "RailroadProfile",
    "import_legacy_rules",
    "import_legacy_tasks",
]
