from __future__ import annotations

from collections import defaultdict

import pytest

from volume2gym.models import AnswerKey, Citation, Task, TaskFamily
from volume2gym.splitter import GroupedSplitter, SplitRatios


def _task(task_id: str, family: str, *unit_ids: str) -> Task:
    citation = Citation(document_id="manual", span_id=f"span-{task_id}")
    return Task(
        task_id=task_id,
        prompt=f"Apply {family}.",
        task_family=TaskFamily("standard_operation"),
        knowledge_unit_ids=unit_ids,
        answer_key=AnswerKey(
            applicable_unit_ids=unit_ids,
            reference_answer="Comply.",
            citations=(citation,),
        ),
        citations=(citation,),
        rule_family=family,
        generator="test",
    )


def test_rule_families_are_leakage_free_and_deterministic() -> None:
    tasks = [
        _task(f"{family}-{variant}", family, f"unit-{family}-{variant}")
        for family in (f"family-{index}" for index in range(9))
        for variant in range(2)
    ]
    splitter = GroupedSplitter(SplitRatios(0.6, 0.2, 0.2), seed="release-1")

    assigned = splitter.split(tasks)
    reversed_assigned = splitter.split(reversed(tasks))

    by_family: dict[str, set[str]] = defaultdict(set)
    for task in assigned:
        assert task.split is not None
        by_family[task.rule_family].add(task.split.value)
    assert all(len(names) == 1 for names in by_family.values())

    first_map = {task.task_id: task.split for task in assigned}
    second_map = {task.task_id: task.split for task in reversed_assigned}
    assert first_map == second_map

    group_counts: dict[str, int] = defaultdict(int)
    for names in by_family.values():
        group_counts[next(iter(names))] += 1
    assert group_counts == {"train": 5, "dev": 2, "test": 2}


def test_knowledge_unit_mode_joins_overlapping_multi_unit_tasks() -> None:
    tasks = [
        _task("a", "one", "A"),
        _task("bridge", "bridge", "A", "B"),
        _task("b", "two", "B"),
        _task("c", "three", "C"),
        _task("d", "four", "D"),
    ]

    assigned = GroupedSplitter(seed=7, group_by="knowledge_unit").split(tasks)
    split_by_id = {task.task_id: task.split for task in assigned}

    assert split_by_id["a"] == split_by_id["bridge"] == split_by_id["b"]


def test_partition_includes_empty_splits() -> None:
    partitions = GroupedSplitter().partition([_task("only", "only-family", "only-unit")])

    assert {split.value for split in partitions} == {"train", "dev", "test"}
    assert sum(len(items) for items in partitions.values()) == 1


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        SplitRatios(1.1, 0.0, -0.1)
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(0.7, 0.2, 0.2)
