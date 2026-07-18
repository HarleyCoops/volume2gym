"""Deterministic, grouped train/dev/test assignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from math import floor, isclose
from typing import Literal

from .models import Split, Task

GroupMode = Literal["rule_family", "knowledge_unit"]
GroupFunction = Callable[[Task], Hashable]


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Target ratios measured in indivisible groups, not individual tasks."""

    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("split ratios must each be between 0 and 1")
        if not isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to 1")


class GroupedSplitter:
    """Assign complete semantic groups to deterministic dataset splits.

    ``rule_family`` holds out entire conceptual families. ``knowledge_unit``
    groups tasks by connected components of their source-unit references, so a
    multi-unit task cannot bridge otherwise separate splits. A callable may be
    supplied for chapter-, document-, or organization-specific grouping.
    """

    def __init__(
        self,
        ratios: SplitRatios | None = None,
        *,
        seed: str | int = 0,
        group_by: GroupMode | GroupFunction = "rule_family",
    ) -> None:
        self.ratios = ratios or SplitRatios()
        self.seed = str(seed)
        self.group_by = group_by

    def split(self, tasks: Iterable[Task]) -> list[Task]:
        """Return copies of tasks with grouped split assignments."""

        records = list(tasks)
        if not records:
            return []

        keys = self._group_keys(records)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            groups[key].append(index)

        assignments = self._assign_groups(groups)
        return [
            _copy_with_split(task, assignments[key])
            for task, key in zip(records, keys, strict=True)
        ]

    def partition(self, tasks: Iterable[Task]) -> dict[Split, tuple[Task, ...]]:
        """Split tasks and return all three partitions, including empty ones."""

        assigned = self.split(tasks)
        result: dict[Split, list[Task]] = {_split(name): [] for name in ("train", "dev", "test")}
        for task in assigned:
            if task.split is None:  # pragma: no cover - protected by split()
                raise RuntimeError(f"task {task.task_id!r} was not assigned a split")
            result[task.split].append(task)
        return {name: tuple(items) for name, items in result.items()}

    def group_assignments(self, tasks: Iterable[Task]) -> dict[str, Split]:
        """Expose the stable group-to-split map for manifests and audits."""

        records = list(tasks)
        keys = self._group_keys(records)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            groups[key].append(index)
        return self._assign_groups(groups)

    def _group_keys(self, tasks: list[Task]) -> list[str]:
        if self.group_by == "knowledge_unit":
            return _knowledge_component_keys(tasks)
        if self.group_by == "rule_family":
            return [
                f"family:{task.rule_family}"
                if task.rule_family
                else f"unit:{'|'.join(sorted(task.knowledge_unit_ids)) or task.task_id}"
                for task in tasks
            ]
        if callable(self.group_by):
            keys = []
            for task in tasks:
                raw = self.group_by(task)
                if raw is None:
                    raise ValueError(f"group function returned None for task {task.task_id!r}")
                keys.append(f"custom:{raw}")
            return keys
        raise ValueError("group_by must be 'rule_family', 'knowledge_unit', or a callable")

    def _assign_groups(self, groups: dict[str, list[int]]) -> dict[str, Split]:
        ordered_groups = sorted(groups, key=self._digest)
        counts = _target_counts(len(ordered_groups), self.ratios)
        boundaries = (counts[0], counts[0] + counts[1])

        assignments: dict[str, Split] = {}
        for position, group in enumerate(ordered_groups):
            if position < boundaries[0]:
                name = _split("train")
            elif position < boundaries[1]:
                name = _split("dev")
            else:
                name = _split("test")
            assignments[group] = name
        return assignments

    def _digest(self, group: str) -> str:
        return sha256(f"{self.seed}\0{group}".encode()).hexdigest()


def split_tasks(
    tasks: Iterable[Task],
    *,
    ratios: SplitRatios | None = None,
    seed: str | int = 0,
    group_by: GroupMode | GroupFunction = "rule_family",
) -> list[Task]:
    """Convenience wrapper around :class:`GroupedSplitter`."""

    return GroupedSplitter(ratios, seed=seed, group_by=group_by).split(tasks)


def _target_counts(group_count: int, ratios: SplitRatios) -> tuple[int, int, int]:
    values = (ratios.train, ratios.dev, ratios.test)
    raw = [group_count * value for value in values]
    counts = [floor(value) for value in raw]
    remaining = group_count - sum(counts)
    remainder_order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in remainder_order[:remaining]:
        counts[index] += 1

    positive = [index for index, ratio in enumerate(values) if ratio > 0]
    if group_count >= len(positive):
        for recipient in positive:
            if counts[recipient] > 0:
                continue
            donors = sorted(
                (index for index in positive if counts[index] > 1),
                key=lambda index: (-counts[index], index),
            )
            if donors:
                counts[donors[0]] -= 1
                counts[recipient] += 1
    return tuple(counts)  # type: ignore[return-value]


def _knowledge_component_keys(tasks: list[Task]) -> list[str]:
    """Compute connected components of overlapping knowledge-unit references."""

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for task in tasks:
        identifiers = tuple(sorted(set(task.knowledge_unit_ids)))
        for identifier in identifiers:
            find(identifier)
        for identifier in identifiers[1:]:
            union(identifiers[0], identifier)

    members: dict[str, list[str]] = defaultdict(list)
    for identifier in parent:
        members[find(identifier)].append(identifier)
    names = {root: "units:" + "|".join(sorted(component)) for root, component in members.items()}

    result = []
    for task in tasks:
        if not task.knowledge_unit_ids:
            result.append(f"task:{task.task_id}")
        else:
            result.append(names[find(task.knowledge_unit_ids[0])])
    return result


def _copy_with_split(task: Task, split: Split) -> Task:
    if hasattr(task, "model_copy"):
        return task.model_copy(update={"split": split})
    return task.copy(update={"split": split})  # pragma: no cover - Pydantic v1 compatibility


def _split(value: str) -> Split:
    return Split(value)


__all__ = ["GroupedSplitter", "SplitRatios", "split_tasks"]
