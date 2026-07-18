"""Domain profiles for importing structured volumes into core contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..models import KnowledgeUnit, Task


@dataclass(frozen=True, slots=True)
class ImportedVolume:
    """Normalized result of importing one legacy or domain-specific bundle."""

    knowledge_units: tuple[KnowledgeUnit, ...]
    tasks: tuple[Task, ...] = ()


@runtime_checkable
class VolumeProfile(Protocol):
    """Minimal adapter contract for domain-specific legacy formats."""

    profile_id: str

    def import_units(
        self,
        payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        document_id: str,
    ) -> list[KnowledgeUnit]: ...

    def import_tasks(
        self,
        payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        units: Sequence[KnowledgeUnit] = (),
        document_id: str,
    ) -> list[Task]: ...


from .railroad import Railroad1959Profile, RailroadProfile  # noqa: E402

__all__ = ["ImportedVolume", "Railroad1959Profile", "RailroadProfile", "VolumeProfile"]
