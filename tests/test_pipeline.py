from __future__ import annotations

import json
from collections import defaultdict

import pytest

from volume2gym.models import Citation, KnowledgeKind, KnowledgeUnit
from volume2gym.pipeline import compile_build, inspect_build, load_build, validate_build


def _unit(unit_id: str, family: str) -> KnowledgeUnit:
    citation = Citation(
        document_id="mini-manual",
        span_id=f"mini-manual:{unit_id}",
        quote=f"Text for {unit_id}.",
    )
    return KnowledgeUnit(
        unit_id=unit_id,
        kind=KnowledgeKind.RULE,
        title=f"Rule {unit_id}",
        text=f"Text for {unit_id}.",
        family=family,
        citations=(citation,),
        required_actions=(f"Apply {unit_id}.",),
        forbidden_actions=(f"Ignore {unit_id}.",),
        procedure_steps=("Recognize.", "Act."),
        terms=(unit_id,),
    )


def test_canonical_units_compile_to_deterministic_complete_build(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    units = [_unit("A", "alpha"), _unit("B", "beta"), _unit("C", "gamma")]
    units_path.write_text(
        "".join(json.dumps(unit.model_dump(mode="json")) + "\n" for unit in units),
        encoding="utf-8",
    )

    first = compile_build(
        volume_id="mini-volume",
        output_dir=tmp_path / "build-one",
        canonical_units_path=units_path,
        seed=1959,
    )
    second = compile_build(
        volume_id="mini-volume",
        output_dir=tmp_path / "build-two",
        canonical_units_path=units_path,
        seed=1959,
    )

    assert len(first.knowledge_units) == 3
    assert len(first.tasks) == 18
    assert first.manifest == second.manifest
    assert first.manifest_ref.sha256 == second.manifest_ref.sha256
    assert validate_build(first.root)["valid"] is True
    assert sum(validate_build(first.root)["split_counts"].values()) == 18

    by_family: dict[str, set[str]] = defaultdict(set)
    for task in first.tasks:
        assert task.split is not None
        by_family[task.rule_family].add(task.split.value)
    assert all(len(splits) == 1 for splits in by_family.values())


def test_build_manifest_is_portable_across_input_directories(tmp_path) -> None:
    first_input = tmp_path / "first" / "units.jsonl"
    second_input = tmp_path / "second" / "units.jsonl"
    first_input.parent.mkdir()
    second_input.parent.mkdir()
    payload = json.dumps(_unit("A", "alpha").model_dump(mode="json")) + "\n"
    first_input.write_text(payload, encoding="utf-8")
    second_input.write_text(payload, encoding="utf-8")

    first = compile_build(
        volume_id="portable-volume",
        output_dir=tmp_path / "build-one",
        canonical_units_path=first_input,
    )
    second = compile_build(
        volume_id="portable-volume",
        output_dir=tmp_path / "build-two",
        canonical_units_path=second_input,
    )

    assert first.manifest == second.manifest
    assert first.manifest_ref.sha256 == second.manifest_ref.sha256
    assert first.manifest.inputs[0].path == "units.jsonl"


def test_legacy_railroad_build_imports_every_rule_and_optional_task(tmp_path) -> None:
    rules = {
        "rules": [
            {
                "rule_id": "7",
                "text": "Employees must know the location of emergency equipment.",
                "category": "duties",
                "page_number": 4,
            },
            {
                "rule_id": "42",
                "text": "Display the prescribed signal before movement.",
                "category": "signals",
                "page_number": 18,
            },
        ]
    }
    tasks = {
        "tasks": [
            {
                "task_id": "legacy-42",
                "description": "What must happen before movement?",
                "applicable_rules": ["42"],
                "expected_outcome": "Display the prescribed signal.",
            }
        ]
    }
    rules_path = tmp_path / "rules.json"
    tasks_path = tmp_path / "tasks.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

    result = compile_build(
        volume_id="railroad-1959",
        output_dir=tmp_path / "railroad-build",
        railroad_rules_path=rules_path,
        railroad_tasks_path=tasks_path,
        seed=42,
    )

    assert len(result.knowledge_units) == 2
    assert len(result.tasks) == 13
    assert {unit.metadata["legacy_rule_id"] for unit in result.knowledge_units} == {"7", "42"}
    assert "legacy-42" in {task.task_id for task in result.tasks}
    assert result.manifest.metadata["generated_task_count"] == 12
    assert result.manifest.metadata["imported_task_count"] == 1
    assert validate_build(result.root)["task_count"] == 13


def test_validation_detects_modified_artifact(tmp_path) -> None:
    result = compile_build(
        volume_id="tamper-test",
        output_dir=tmp_path / "build",
        units=[_unit("A", "alpha")],
    )
    tasks_path = result.root / "tasks" / "all.jsonl"
    tasks_path.write_text(tasks_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_build(result.root)


def test_inspection_reports_artifacts_and_task_families(tmp_path) -> None:
    result = compile_build(
        volume_id="inspect-test",
        output_dir=tmp_path / "build",
        units=[_unit("A", "alpha")],
    )

    summary = inspect_build(result.root)
    loaded = load_build(result.root)

    assert summary["valid"] is True
    assert summary["artifact_count"] == 5
    assert len(summary["artifacts"]) == 5
    assert set(summary["task_family_counts"]) == {
        "standard_operation",
        "edge_case",
        "conflict_resolution",
        "exception_handling",
        "violation_check",
        "adversarial_distractor",
    }
    assert loaded.manifest.build_id == result.manifest.build_id
