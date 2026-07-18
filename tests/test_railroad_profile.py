from __future__ import annotations

import pytest

from volume2gym.compiler import TemplateCompiler
from volume2gym.profiles import ImportedVolume, RailroadProfile, VolumeProfile

RULES = {
    "rules": [
        {
            "rule_id": "7",
            "text": "Employees must know where the emergency equipment is kept.",
            "category": "General Duties",
            "section": "Safety",
            "page_number": 4,
            "sub_rules": [
                {
                    "rule_id": "7A",
                    "text": "Report missing emergency equipment immediately.",
                }
            ],
        },
        {
            "rule_id": "42",
            "text": "Display the prescribed signal before movement.",
            "category": "Signals",
            "page_number": 18,
        },
    ]
}

TASKS = {
    "tasks": [
        {
            "task_id": "7-001",
            "description": "Emergency equipment is missing. What is the required response?",
            "applicable_rules": ["7", "7A"],
            "expected_outcome": "Report the missing equipment immediately.",
            "task_type": "violation",
        },
        {
            "task_id": "42-001",
            "description": "A movement is ready to begin. What must happen first?",
            "applicable_rules": ["42"],
            "expected_outcome": "Display the prescribed signal before movement.",
        },
    ]
}


def test_profile_normalizes_all_rules_and_nested_sub_rules() -> None:
    profile = RailroadProfile()

    units = profile.import_units(RULES, document_id="code-1959")

    assert isinstance(profile, VolumeProfile)
    assert [unit.metadata["legacy_rule_id"] for unit in units] == ["7", "7A", "42"]
    by_legacy_id = {unit.metadata["legacy_rule_id"]: unit for unit in units}
    assert by_legacy_id["7A"].family == "General Duties"
    assert by_legacy_id["7A"].section == "Safety"
    assert by_legacy_id["7A"].citations[0].page_number == 4
    assert by_legacy_id["7A"].metadata["parent_unit_id"] == by_legacy_id["7"].unit_id
    assert by_legacy_id["7"].citations[0].page_number == 4
    assert by_legacy_id["7"].citations[0].document_id == "code-1959"
    assert by_legacy_id["42"].unit_id == "railroad-rule-42"


def test_bundle_maps_legacy_task_rules_to_normalized_units() -> None:
    bundle = RailroadProfile().import_bundle(RULES, TASKS, document_id="code-1959")

    assert isinstance(bundle, ImportedVolume)
    assert len(bundle.knowledge_units) == 3
    assert len(bundle.tasks) == 2
    units = {unit.metadata["legacy_rule_id"]: unit for unit in bundle.knowledge_units}
    tasks = {task.task_id: task for task in bundle.tasks}

    assert tasks["7-001"].knowledge_unit_ids == (units["7"].unit_id, units["7A"].unit_id)
    assert tasks["7-001"].answer_key.applicable_unit_ids == tasks["7-001"].knowledge_unit_ids
    assert tasks["7-001"].task_family.value == "violation_check"
    assert tasks["42-001"].task_family.value == "applied_scenario"
    assert tasks["42-001"].answer_key.reference_answer == TASKS["tasks"][1]["expected_outcome"]


def test_imported_units_compile_without_any_rule_number_special_case() -> None:
    units = RailroadProfile().import_units(RULES)

    compiled = TemplateCompiler().compile(units)

    assert len(compiled) == 18
    assert {task.knowledge_unit_ids[0] for task in compiled} == {unit.unit_id for unit in units}


def test_duplicate_or_normalization_colliding_rule_ids_fail_loudly() -> None:
    payload = [
        {"rule_id": "A B", "text": "First."},
        {"rule_id": "A-B", "text": "Second."},
    ]

    with pytest.raises(ValueError, match="duplicate or colliding"):
        RailroadProfile().import_units(payload)


def test_task_without_applicable_rules_is_rejected() -> None:
    with pytest.raises(ValueError, match="has no applicable_rules"):
        RailroadProfile().import_tasks(
            [
                {
                    "task_id": "broken",
                    "description": "What now?",
                    "expected_outcome": "Unknown.",
                }
            ]
        )
