"""Versioned, source-grounded artifact contracts used by every Volume2Gym stage."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1"


class V2GModel(BaseModel):
    """Immutable base for persistent build artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    schema_version: Literal["0.1"] = SCHEMA_VERSION


class RightsStatus(StrEnum):
    UNKNOWN = "unknown"
    CLEARED = "cleared"
    RESTRICTED = "restricted"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class KnowledgeKind(StrEnum):
    RULE = "rule"
    PROCEDURE = "procedure"
    DEFINITION = "definition"
    CONCEPT = "concept"
    TABLE = "table"
    EXAMPLE = "example"


class TaskFamily(StrEnum):
    STANDARD_OPERATION = "standard_operation"
    EDGE_CASE = "edge_case"
    VIOLATION_CHECK = "violation_check"
    CONFLICT_RESOLUTION = "conflict_resolution"
    EXCEPTION_HANDLING = "exception_handling"
    ADVERSARIAL_DISTRACTOR = "adversarial_distractor"
    APPLIED_SCENARIO = "applied_scenario"


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class FailureSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RightsInfo(V2GModel):
    status: RightsStatus = RightsStatus.UNKNOWN
    license: str | None = None
    redistributable: bool | None = None
    notes: str | None = None


class SourceDocument(V2GModel):
    document_id: str
    title: str
    edition: str | None = None
    source_uri: str | None = None
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights: RightsInfo = Field(default_factory=RightsInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSpan(V2GModel):
    span_id: str
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def populate_text_hash(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("text") and not value.get("text_sha256"):
            value = dict(value)
            value["text_sha256"] = hashlib.sha256(value["text"].encode("utf-8")).hexdigest()
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> SourceSpan:
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        return self


class Citation(V2GModel):
    document_id: str
    span_id: str
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    locator: str | None = None
    quote: str | None = None
    quote_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def populate_quote_hash(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("quote") and not value.get("quote_hash"):
            value = dict(value)
            value["quote_hash"] = hashlib.sha256(value["quote"].encode("utf-8")).hexdigest()
        return value


class KnowledgeUnit(V2GModel):
    unit_id: str
    kind: KnowledgeKind = KnowledgeKind.CONCEPT
    title: str
    text: str
    family: str | None = None
    section: str | None = None
    citations: tuple[Citation, ...]
    conditions: tuple[str, ...] = ()
    required_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    procedure_steps: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    related_unit_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: tuple[Citation, ...]) -> tuple[Citation, ...]:
        if not value:
            raise ValueError("knowledge units must retain at least one source citation")
        return value

    @property
    def canonical_text(self) -> str:
        return self.text

    @property
    def section_path(self) -> tuple[str, ...]:
        return (self.section,) if self.section else ()


DEFAULT_ANSWER_FIELDS = (
    "applicable_rules",
    "situation_type",
    "required_actions",
    "forbidden_actions",
    "procedure_order",
    "final_answer",
)


class AnswerContract(V2GModel):
    format: str = "structured_answer_json"
    fields: tuple[str, ...] = DEFAULT_ANSWER_FIELDS


class AnswerKey(V2GModel):
    applicable_unit_ids: tuple[str, ...]
    required_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    procedure_order: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    reference_answer: str | None = None
    citations: tuple[Citation, ...] = ()


class Task(V2GModel):
    task_id: str
    prompt: str
    task_family: TaskFamily
    knowledge_unit_ids: tuple[str, ...]
    answer_contract: AnswerContract = Field(default_factory=AnswerContract)
    answer_key: AnswerKey
    citations: tuple[Citation, ...] = ()
    rule_family: str | None = None
    tags: tuple[str, ...] = ()
    generator: str = "unspecified"
    split: Split | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_grounding(self) -> Task:
        if not self.knowledge_unit_ids:
            raise ValueError("tasks must reference at least one knowledge unit")
        if set(self.answer_key.applicable_unit_ids) - set(self.knowledge_unit_ids):
            raise ValueError("answer key references a knowledge unit absent from the task")
        return self

    @property
    def task_type(self) -> TaskFamily:
        return self.task_family


TaskSpec = Task
TaskType = TaskFamily


class StructuredAnswer(V2GModel):
    applicable_rules: tuple[str, ...] = ()
    situation_type: str | None = None
    required_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    procedure_order: tuple[str, ...] = ()
    final_answer: str = ""


class ModelResponse(V2GModel):
    response_id: str
    task_id: str
    raw_text: str
    structured_answer: StructuredAnswer | None = None
    parse_error: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardComponent(V2GModel):
    component_id: str
    weight: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    weighted_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    notes: str | None = None
    verifier_id: str = "deterministic-contract"

    @model_validator(mode="before")
    @classmethod
    def populate_weighted_score(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("weighted_score") is None:
            value = dict(value)
            value["weighted_score"] = float(value.get("weight", 0.0)) * float(
                value.get("score", 0.0)
            )
        return value

    @model_validator(mode="after")
    def reconcile(self) -> RewardComponent:
        expected = self.weight * self.score
        if self.weighted_score is None or not math.isclose(
            self.weighted_score, expected, abs_tol=1e-9
        ):
            raise ValueError("weighted_score must equal weight * score")
        return self


class RewardLedger(V2GModel):
    task_id: str
    response_id: str
    components: tuple[RewardComponent, ...]
    ungated_total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    gate_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    verifier_id: str
    verifier_version: str

    @model_validator(mode="before")
    @classmethod
    def populate_totals(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        components = value.get("components") or ()
        raw = 0.0
        for component in components:
            if isinstance(component, RewardComponent):
                raw += component.weighted_score or 0.0
            elif isinstance(component, dict):
                weighted = component.get("weighted_score")
                if weighted is None:
                    weighted = float(component.get("weight", 0.0)) * float(
                        component.get("score", 0.0)
                    )
                raw += float(weighted)
        if math.isclose(raw, 1.0, abs_tol=1e-9):
            raw = 1.0
        value.setdefault("ungated_total_score", raw)
        total = raw * float(value.get("gate_multiplier", 1.0))
        if math.isclose(total, 1.0, abs_tol=1e-9):
            total = 1.0
        value.setdefault("total_score", total)
        return value

    @model_validator(mode="after")
    def reconcile(self) -> RewardLedger:
        if not self.components:
            raise ValueError("reward ledgers require at least one component")
        weight_sum = sum(component.weight for component in self.components)
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
            raise ValueError("reward component weights must sum to 1")
        calculated = sum(component.weighted_score or 0.0 for component in self.components)
        if self.ungated_total_score is None or not math.isclose(
            self.ungated_total_score, calculated, abs_tol=1e-9
        ):
            raise ValueError("ungated_total_score must equal weighted component scores")
        expected_total = calculated * self.gate_multiplier
        if self.total_score is None or not math.isclose(
            self.total_score, expected_total, abs_tol=1e-9
        ):
            raise ValueError("total_score must equal ungated_total_score * gate_multiplier")
        return self

    @property
    def component_map(self) -> dict[str, RewardComponent]:
        return {component.component_id: component for component in self.components}


class EvaluationRecord(V2GModel):
    task_id: str
    response: ModelResponse
    reward_ledger: RewardLedger


class EvaluationReport(V2GModel):
    evaluation_id: str
    records: tuple[EvaluationRecord, ...]
    mean_score: float = Field(ge=0.0, le=1.0)
    split: Split | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureCluster(V2GModel):
    cluster_id: str
    component: str
    count: int = Field(ge=1)
    task_ids: tuple[str, ...]
    knowledge_unit_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    severity: FailureSeverity = FailureSeverity.WARNING
    signature: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CurriculumRequest(V2GModel):
    request_id: str
    failure_cluster_id: str
    component: str
    goal: str
    task_family: TaskFamily
    knowledge_unit_ids: tuple[str, ...]
    parent_task_ids: tuple[str, ...]
    citations: tuple[Citation, ...] = ()
    difficulty: str = "targeted"
    priority: int = Field(default=1, ge=1)
    expected_required_actions: tuple[str, ...] = ()
    unsafe_counterexamples: tuple[str, ...] = ()

    @property
    def target_component(self) -> str:
        return self.component

    @property
    def source_task_ids(self) -> tuple[str, ...]:
        return self.parent_task_ids


class ArtifactRef(V2GModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    record_count: int | None = Field(default=None, ge=0)


class BuildManifest(V2GModel):
    build_id: str
    volume_id: str
    compiler_version: str
    seed: int
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    source_revision: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainerRecipe(V2GModel):
    trainer_id: str
    method: str
    base_model: str | None = None
    seed: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelArtifact(V2GModel):
    model_id: str
    artifact_type: str
    uri: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recipe: TrainerRecipe
    metadata: dict[str, Any] = Field(default_factory=dict)


class CycleManifest(V2GModel):
    cycle_id: str
    build_id: str
    parent_cycle_id: str | None = None
    model: ModelArtifact | None = None
    evaluation: ArtifactRef | None = None
    failures: tuple[FailureCluster, ...] = ()
    next_curriculum: tuple[CurriculumRequest, ...] = ()


class Observation(V2GModel):
    task_id: str
    prompt: str
    answer_contract: AnswerContract


__all__ = [
    "AnswerContract",
    "AnswerKey",
    "ArtifactRef",
    "BuildManifest",
    "Citation",
    "CurriculumRequest",
    "CycleManifest",
    "EvaluationRecord",
    "EvaluationReport",
    "FailureCluster",
    "FailureSeverity",
    "KnowledgeKind",
    "KnowledgeUnit",
    "ModelArtifact",
    "ModelResponse",
    "Observation",
    "ReviewStatus",
    "RewardComponent",
    "RewardLedger",
    "RightsInfo",
    "RightsStatus",
    "SCHEMA_VERSION",
    "SourceDocument",
    "SourceSpan",
    "Split",
    "StructuredAnswer",
    "Task",
    "TaskFamily",
    "TaskSpec",
    "TaskType",
    "TrainerRecipe",
    "V2GModel",
]
