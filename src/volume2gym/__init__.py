"""Public API for compiling structured volumes into learning environments.

The top-level package intentionally exposes only dependency-light contracts
and the complete local compilation loop. Provider and framework integrations
remain available from their explicit submodules.
"""

from .artifacts import ArtifactStore
from .compiler import TemplateCompiler, compile_units
from .environment import VolumeGym
from .exporters import (
    grpo_record,
    reference_answer,
    sft_record,
    write_grpo_jsonl,
    write_sft_jsonl,
)
from .extraction import (
    ExtractionError,
    StructuredKnowledgeExtractor,
    extract_knowledge_units,
)
from .failures import build_curriculum, cluster_failures
from .models import (
    SCHEMA_VERSION,
    AnswerContract,
    AnswerKey,
    ArtifactRef,
    BuildManifest,
    Citation,
    CurriculumRequest,
    CycleManifest,
    FailureCluster,
    KnowledgeKind,
    KnowledgeUnit,
    ModelArtifact,
    ModelResponse,
    RewardComponent,
    RewardLedger,
    SourceDocument,
    SourceSpan,
    Split,
    StructuredAnswer,
    Task,
    TaskFamily,
    TrainerRecipe,
)
from .pipeline import BuildResult, compile_build, inspect_build, validate_build
from .splitter import GroupedSplitter, split_tasks
from .trainers import SymbolicPolicy, SymbolicTrainer, evaluate_policy
from .verifier import DeterministicVerifier, verify_answer

__version__ = "0.1.0"

__all__ = [
    "AnswerContract",
    "AnswerKey",
    "ArtifactRef",
    "ArtifactStore",
    "BuildManifest",
    "BuildResult",
    "Citation",
    "CurriculumRequest",
    "CycleManifest",
    "DeterministicVerifier",
    "ExtractionError",
    "FailureCluster",
    "GroupedSplitter",
    "KnowledgeKind",
    "KnowledgeUnit",
    "ModelArtifact",
    "ModelResponse",
    "RewardComponent",
    "RewardLedger",
    "SCHEMA_VERSION",
    "SourceDocument",
    "SourceSpan",
    "Split",
    "StructuredAnswer",
    "StructuredKnowledgeExtractor",
    "SymbolicPolicy",
    "SymbolicTrainer",
    "Task",
    "TaskFamily",
    "TemplateCompiler",
    "TrainerRecipe",
    "VolumeGym",
    "__version__",
    "build_curriculum",
    "cluster_failures",
    "compile_build",
    "compile_units",
    "evaluate_policy",
    "extract_knowledge_units",
    "grpo_record",
    "inspect_build",
    "reference_answer",
    "sft_record",
    "split_tasks",
    "validate_build",
    "verify_answer",
    "write_grpo_jsonl",
    "write_sft_jsonl",
]
