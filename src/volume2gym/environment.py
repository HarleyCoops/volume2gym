"""A dependency-light, single-turn environment over compiled volume tasks."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from .models import ModelResponse, RewardLedger, StructuredAnswer, Task
from .verifier import DeterministicVerifier

Action = StructuredAnswer | ModelResponse | Mapping[str, Any] | str


class VolumeGym:
    """Present one compiled task and terminate after one scored answer.

    ``VolumeGym`` follows the familiar Gymnasium ``reset``/``step`` return
    shapes without requiring Gymnasium in the core package.  Sampling uses a
    private RNG, so ``reset(seed=...)`` is reproducible and cannot disturb
    process-global randomness.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        tasks: Iterable[Task],
        *,
        verifier: DeterministicVerifier | None = None,
        seed: int | str | bytes | bytearray | None = None,
    ) -> None:
        self._tasks = tuple(tasks)
        if not self._tasks:
            raise ValueError("VolumeGym requires at least one task")
        identifiers = [task.task_id for task in self._tasks]
        duplicates = sorted({task_id for task_id in identifiers if identifiers.count(task_id) > 1})
        if duplicates:
            raise ValueError("duplicate task ids: " + ", ".join(duplicates))

        self._task_by_id = {task.task_id: task for task in self._tasks}
        self.verifier = verifier or DeterministicVerifier()
        self._rng = random.Random(seed)
        self._current_task: Task | None = None
        self._terminated = False
        self._episode_index = 0
        self._last_ledger: RewardLedger | None = None

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self._tasks

    @property
    def current_task(self) -> Task | None:
        return self._current_task

    @property
    def last_ledger(self) -> RewardLedger | None:
        return self._last_ledger

    def reset(
        self,
        *,
        seed: int | str | bytes | bytearray | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Start an episode, optionally selecting an exact ``task_id``."""

        if seed is not None:
            self._rng.seed(seed)
        requested_id = (options or {}).get("task_id")
        if requested_id is None:
            task = self._rng.choice(self._tasks)
        else:
            try:
                task = self._task_by_id[str(requested_id)]
            except KeyError as exc:
                raise KeyError(f"unknown task_id: {requested_id}") from exc

        self._current_task = task
        self._terminated = False
        self._last_ledger = None
        self._episode_index += 1
        return task.prompt, self._task_info(task)

    def step(self, action: Action) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Score ``action`` and terminate the current episode."""

        if self._current_task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset() before another step()")

        task = self._current_task
        response_id = (
            action.response_id
            if isinstance(action, ModelResponse)
            else f"{task.task_id}.episode-{self._episode_index}.response"
        )
        ledger = self.verifier.verify(task, action, response_id=response_id)
        self._last_ledger = ledger
        self._terminated = True

        info = self._task_info(task)
        info["reward_ledger"] = ledger.model_dump(mode="json")
        return "Task complete", ledger.total_score, True, False, info

    def render(self) -> str:
        if self._current_task is None:
            return "VolumeGym: no active task"
        state = "terminated" if self._terminated else "awaiting response"
        return f"[{self._current_task.task_id}] {state}\n{self._current_task.prompt}"

    def close(self) -> None:
        """Release the active episode state; retained tasks are immutable."""

        self._current_task = None
        self._terminated = False
        self._last_ledger = None

    @staticmethod
    def _task_info(task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "task_family": task.task_family.value,
            "knowledge_unit_ids": list(task.knowledge_unit_ids),
            "rule_family": task.rule_family,
            "split": task.split.value if task.split is not None else None,
            "citations": [citation.model_dump(mode="json") for citation in task.citations],
        }


SingleTurnVolumeGym = VolumeGym


__all__ = ["Action", "SingleTurnVolumeGym", "VolumeGym"]
