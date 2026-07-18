"""Trainer-neutral policies plus a transparent symbolic reference trainer."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from volume2gym.exporters import reference_answer
from volume2gym.models import EvaluationRecord, ModelResponse, StructuredAnswer, Task
from volume2gym.verifier import DeterministicVerifier


class SymbolicPolicy:
    """An inspectable non-neural policy used to test the full artifact loop."""

    model_id = "volume2gym-symbolic-reference"

    def __init__(self, answers: dict[str, StructuredAnswer]) -> None:
        self._answers = dict(answers)

    def respond(self, task: Task) -> ModelResponse:
        try:
            answer = self._answers[task.task_id]
        except KeyError as exc:
            raise KeyError(f"symbolic policy has no answer for {task.task_id}") from exc
        digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:16]
        return ModelResponse(
            response_id=f"symbolic-{digest}",
            task_id=task.task_id,
            raw_text=answer.model_dump_json(),
            structured_answer=answer,
            model_id=self.model_id,
        )


class SymbolicTrainer:
    """Compile answer contracts into a deterministic policy; never label it neural."""

    trainer_id = "volume2gym-symbolic-trainer"

    def train(self, tasks: Iterable[Task]) -> SymbolicPolicy:
        answers: dict[str, StructuredAnswer] = {}
        for task in tasks:
            if task.task_id in answers:
                raise ValueError(f"duplicate task id: {task.task_id}")
            answers[task.task_id] = reference_answer(task)
        if not answers:
            raise ValueError("symbolic training requires at least one task")
        return SymbolicPolicy(answers)


def evaluate_policy(
    policy: SymbolicPolicy,
    tasks: Iterable[Task],
    *,
    verifier: DeterministicVerifier | None = None,
) -> tuple[EvaluationRecord, ...]:
    scorer = verifier or DeterministicVerifier()
    records: list[EvaluationRecord] = []
    for task in tasks:
        response = policy.respond(task)
        records.append(
            EvaluationRecord(
                task_id=task.task_id,
                response=response,
                reward_ledger=scorer.verify(task, response),
            )
        )
    return tuple(records)


__all__ = ["SymbolicPolicy", "SymbolicTrainer", "evaluate_policy"]
