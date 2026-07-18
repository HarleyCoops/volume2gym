"""Stable boundaries between the compiler, providers, gyms, and trainers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    ArtifactRef,
    BuildManifest,
    FailureCluster,
    KnowledgeUnit,
    ModelArtifact,
    ModelResponse,
    RewardLedger,
    SourceDocument,
    SourceSpan,
    Split,
    Task,
    TrainerRecipe,
)


@runtime_checkable
class SourceReader(Protocol):
    """Read a declared source without losing page/span identity."""

    def read(self, document: SourceDocument) -> Iterable[SourceSpan]: ...


@runtime_checkable
class KnowledgeExtractor(Protocol):
    """Convert source spans into normalized, cited knowledge units."""

    def extract(
        self,
        document: SourceDocument,
        spans: Sequence[SourceSpan],
    ) -> Iterable[KnowledgeUnit]: ...


@runtime_checkable
class TaskGenerator(Protocol):
    """Generate source-grounded tasks without selecting a trainer."""

    def compile(self, units: Iterable[KnowledgeUnit]) -> list[Task]: ...


@runtime_checkable
class TaskSplitter(Protocol):
    def split(self, tasks: Iterable[Task]) -> list[Task]: ...

    def partition(self, tasks: Iterable[Task]) -> Mapping[Split, tuple[Task, ...]]: ...


@runtime_checkable
class StructuredProvider(Protocol):
    """Provider-neutral structured generation contract."""

    provider: str
    model: str

    def generate(self, request: Any) -> Any: ...


@runtime_checkable
class Verifier(Protocol):
    verifier_id: str
    verifier_version: str

    def verify(
        self,
        task: Task,
        response: ModelResponse | Mapping[str, Any] | str,
        *,
        response_id: str | None = None,
    ) -> RewardLedger: ...


@runtime_checkable
class VolumePolicy(Protocol):
    model_id: str

    def respond(self, task: Task) -> ModelResponse: ...


@runtime_checkable
class TrainerAdapter(Protocol):
    trainer_id: str

    def train(
        self,
        build: BuildManifest,
        train_tasks: Sequence[Task],
        dev_tasks: Sequence[Task],
        recipe: TrainerRecipe,
        *,
        output_dir: Path,
    ) -> ModelArtifact: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        policy: VolumePolicy,
        tasks: Sequence[Task],
        verifier: Verifier,
    ) -> Sequence[RewardLedger]: ...


@runtime_checkable
class FailureMiner(Protocol):
    def mine(
        self,
        ledgers: Sequence[RewardLedger],
        tasks: Sequence[Task],
    ) -> Sequence[FailureCluster]: ...


@runtime_checkable
class ArtifactRepository(Protocol):
    root: Path

    def write_json(self, relative_path: str | Path, value: Any) -> ArtifactRef: ...

    def write_jsonl(
        self,
        relative_path: str | Path,
        records: Iterable[Any],
    ) -> ArtifactRef: ...

    def validate(self, reference: ArtifactRef) -> None: ...


__all__ = [
    "ArtifactRepository",
    "Evaluator",
    "FailureMiner",
    "KnowledgeExtractor",
    "SourceReader",
    "StructuredProvider",
    "TaskGenerator",
    "TaskSplitter",
    "TrainerAdapter",
    "Verifier",
    "VolumePolicy",
]
