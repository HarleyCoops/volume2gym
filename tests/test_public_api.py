from __future__ import annotations

import volume2gym
from volume2gym.artifacts import ArtifactStore
from volume2gym.compiler import TemplateCompiler
from volume2gym.environment import VolumeGym
from volume2gym.exporters import write_grpo_jsonl, write_sft_jsonl
from volume2gym.models import KnowledgeUnit, Task
from volume2gym.pipeline import compile_build
from volume2gym.verifier import DeterministicVerifier


def test_public_api_is_explicit_unique_and_resolvable() -> None:
    assert volume2gym.__all__ == sorted(volume2gym.__all__)
    assert len(volume2gym.__all__) == len(set(volume2gym.__all__))
    assert all(hasattr(volume2gym, name) for name in volume2gym.__all__)


def test_core_top_level_exports_are_the_canonical_objects() -> None:
    assert volume2gym.ArtifactStore is ArtifactStore
    assert volume2gym.TemplateCompiler is TemplateCompiler
    assert volume2gym.VolumeGym is VolumeGym
    assert volume2gym.KnowledgeUnit is KnowledgeUnit
    assert volume2gym.Task is Task
    assert volume2gym.compile_build is compile_build
    assert volume2gym.DeterministicVerifier is DeterministicVerifier
    assert volume2gym.write_sft_jsonl is write_sft_jsonl
    assert volume2gym.write_grpo_jsonl is write_grpo_jsonl


def test_importing_public_api_does_not_require_optional_frameworks() -> None:
    # Importing the core must not load SDKs that belong to optional extras.
    import sys

    assert "anthropic" not in sys.modules
    assert "datasets" not in sys.modules
    assert "fitz" not in sys.modules
    assert "gymnasium" not in sys.modules
