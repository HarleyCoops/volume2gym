"""Failure clustering and deterministic next-curriculum planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import (
    CurriculumRequest,
    FailureCluster,
    RewardComponent,
    RewardLedger,
    Task,
    TaskFamily,
)
from .verifier import FORBIDDEN_COMPONENT, SAFETY_COMPONENT

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def cluster_failures(
    ledgers: Iterable[RewardLedger],
    *,
    tasks: Iterable[Task] = (),
    threshold: float = 1.0,
    minimum_count: int = 1,
) -> list[FailureCluster]:
    """Group sub-threshold component scores across reward ledgers.

    Clustering is component-based, matching the reward contract rather than
    relying on free-text similarity.  This keeps results deterministic and
    makes each cluster directly convertible into a curriculum objective.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if minimum_count < 1:
        raise ValueError("minimum_count must be at least 1")

    task_by_id = _task_map(tasks)
    buckets: dict[str, list[tuple[RewardLedger, RewardComponent]]] = defaultdict(list)
    for ledger in ledgers:
        for component in _components(ledger):
            if component.score < threshold:
                buckets[component.component_id].append((ledger, component))

    clusters: list[FailureCluster] = []
    for component_id, failures in buckets.items():
        if len(failures) < minimum_count:
            continue
        task_ids = _unique(ledger.task_id for ledger, _ in failures)
        knowledge_unit_ids = _unique(
            unit_id
            for task_id in task_ids
            if (task := task_by_id.get(task_id)) is not None
            for unit_id in task.knowledge_unit_ids
        )
        evidence = _unique(
            text
            for _, component in failures
            for text in (*component.evidence, *((component.notes,) if component.notes else ()))
        )
        severity = _cluster_severity(component_id, failures)
        clusters.append(
            FailureCluster(
                cluster_id=f"failure:{component_id}",
                component=component_id,
                severity=severity,
                count=len(failures),
                task_ids=task_ids,
                knowledge_unit_ids=knowledge_unit_ids,
                evidence=evidence,
            )
        )
    return rank_failure_clusters(clusters)


def rank_failure_clusters(clusters: Iterable[FailureCluster]) -> list[FailureCluster]:
    """Return clusters in stable safety/severity/frequency priority order."""

    return sorted(
        clusters,
        key=lambda cluster: (
            -_component_priority(cluster.component),
            -_SEVERITY_RANK[_enum_value(cluster.severity)],
            -cluster.count,
            cluster.component,
            cluster.cluster_id,
        ),
    )


def build_curriculum(
    clusters: Iterable[FailureCluster],
    tasks: Iterable[Task],
    *,
    limit: int | None = None,
) -> list[CurriculumRequest]:
    """Turn ranked failures into source-grounded generation objectives."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    task_by_id = _task_map(tasks)
    ranked = rank_failure_clusters(clusters)
    if limit is not None:
        ranked = ranked[:limit]

    targets: list[CurriculumRequest] = []
    for priority, cluster in enumerate(ranked, start=1):
        source_tasks = tuple(
            task_by_id[task_id] for task_id in cluster.task_ids if task_id in task_by_id
        )
        required_actions = _unique(
            action for task in source_tasks for action in task.answer_key.required_actions
        )
        forbidden_actions = _unique(
            action for task in source_tasks for action in task.answer_key.forbidden_actions
        )
        knowledge_unit_ids = cluster.knowledge_unit_ids or _unique(
            unit_id for task in source_tasks for unit_id in task.knowledge_unit_ids
        )
        citations = _unique_models(
            citation for task in source_tasks for citation in task.citations
        )
        task_family = (
            source_tasks[0].task_family if source_tasks else TaskFamily.APPLIED_SCENARIO
        )
        targets.append(
            CurriculumRequest(
                request_id=f"curriculum:{cluster.cluster_id}",
                failure_cluster_id=cluster.cluster_id,
                component=cluster.component,
                goal=_curriculum_instruction(
                    cluster,
                    required_actions=required_actions,
                    forbidden_actions=forbidden_actions,
                ),
                task_family=task_family,
                knowledge_unit_ids=knowledge_unit_ids,
                parent_task_ids=cluster.task_ids,
                citations=citations,
                difficulty="targeted",
                priority=priority,
                expected_required_actions=required_actions,
                unsafe_counterexamples=forbidden_actions,
            )
        )
    return targets


def curriculum_records(
    clusters: Iterable[FailureCluster],
    tasks: Iterable[Task],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-serializable records suitable for a next-batch JSONL."""

    return [
        target.model_dump(mode="json")
        for target in build_curriculum(clusters, tasks, limit=limit)
    ]


def _components(ledger: RewardLedger) -> tuple[RewardComponent, ...]:
    components = ledger.components
    if isinstance(components, Mapping):
        return tuple(components.values())
    return tuple(components)


def _task_map(tasks: Iterable[Task]) -> dict[str, Task]:
    result: dict[str, Task] = {}
    for task in tasks:
        if task.task_id in result:
            raise ValueError(f"duplicate task id: {task.task_id}")
        result[task.task_id] = task
    return result


def _cluster_severity(
    component_id: str,
    failures: Sequence[tuple[RewardLedger, RewardComponent]],
) -> str:
    if component_id == SAFETY_COMPONENT:
        return "critical"
    if component_id == FORBIDDEN_COMPONENT and any(
        float(ledger.diagnostics.get("unsafe_recommendation_detected", 0.0)) > 0
        for ledger, _ in failures
    ):
        return "critical"
    mean_score = sum(component.score for _, component in failures) / len(failures)
    if mean_score <= 0.25:
        return "high"
    if mean_score <= 0.50:
        return "medium"
    return "low"


def _component_priority(component_id: str) -> int:
    if component_id == SAFETY_COMPONENT:
        return 3
    if component_id == FORBIDDEN_COMPONENT:
        return 2
    return 1


def _curriculum_instruction(
    cluster: FailureCluster,
    *,
    required_actions: Sequence[str],
    forbidden_actions: Sequence[str],
) -> str:
    if cluster.component == SAFETY_COMPONENT:
        objective = "practice every mandatory safety action and reject omissions"
    elif cluster.component == FORBIDDEN_COMPONENT:
        objective = "identify prohibited actions and replace unsafe recommendations"
    else:
        objective = f"improve the {cluster.component} reward component"
    instruction = (
        f"Create source-grounded scenarios that {objective}. Preserve the original "
        "citations and structured answer contract, and evaluate them with the same verifier."
    )
    if required_actions:
        instruction += " Required actions: " + "; ".join(required_actions) + "."
    if forbidden_actions:
        instruction += " Unsafe counterexamples: " + "; ".join(forbidden_actions) + "."
    return instruction


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _unique_models(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = value.model_dump_json()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


__all__ = [
    "build_curriculum",
    "cluster_failures",
    "curriculum_records",
    "rank_failure_clusters",
]
