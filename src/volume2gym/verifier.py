"""Deterministic, source-contract verification for structured answers.

The verifier intentionally performs no model calls.  It scores only fields in a
task's answer key against the corresponding fields in a structured answer.  A
component ledger retains the pre-gate scores while a separate multiplier makes
the safety hard gate explicit and auditable.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .models import ModelResponse, RewardComponent, RewardLedger, StructuredAnswer, Task

SAFETY_COMPONENT = "safety_critical_required_actions"
FORBIDDEN_COMPONENT = "forbidden_unsafe_actions_absent"
RULE_COMPONENT = "correct_applicable_rule_cited"
PROCEDURE_COMPONENT = "procedure_order_correct"
TERMINOLOGY_COMPONENT = "domain_terminology_correct"

DEFAULT_WEIGHTS: dict[str, float] = {
    SAFETY_COMPONENT: 0.40,
    FORBIDDEN_COMPONENT: 0.20,
    RULE_COMPONENT: 0.15,
    PROCEDURE_COMPONENT: 0.15,
    TERMINOLOGY_COMPONENT: 0.10,
}

_COMPONENT_ORDER = tuple(DEFAULT_WEIGHTS)
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


class DeterministicVerifier:
    """Score a :class:`Task` without a network call or semantic judge.

    Custom weights are merged over :data:`DEFAULT_WEIGHTS` and normalized to
    sum to one.  When ``safety_hard_gate`` is enabled, missing too many required
    safety actions or positively recommending a forbidden action sets the final
    reward to zero.  Raw component scores remain available in the ledger.
    """

    verifier_id = "volume2gym.deterministic-composite"
    verifier_version = "1"

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        safety_hard_gate: bool = True,
        safety_threshold: float = 1.0,
    ) -> None:
        self.weights = _validated_weights(weights)
        if not 0.0 <= safety_threshold <= 1.0:
            raise ValueError("safety_threshold must be between 0 and 1")
        self.safety_hard_gate = safety_hard_gate
        self.safety_threshold = float(safety_threshold)

    def verify(
        self,
        task: Task,
        response: StructuredAnswer | ModelResponse | Mapping[str, Any] | str,
        *,
        response_id: str | None = None,
    ) -> RewardLedger:
        """Return a reconciled component ledger for one task response."""

        answer, resolved_response_id, schema_validity = _coerce_answer(
            task,
            response,
            response_id=response_id,
        )
        if answer is None:
            return self._invalid_ledger(
                task,
                response_id=resolved_response_id,
                schema_validity=schema_validity,
            )

        expected = task.answer_key
        required_score, matched_required, missing_required = _coverage(
            expected.required_actions,
            answer.required_actions,
        )
        declared_forbidden_score, matched_forbidden, missing_forbidden = _coverage(
            expected.forbidden_actions,
            answer.forbidden_actions,
        )
        unsafe_recommendations = _intersection(
            expected.forbidden_actions,
            (*answer.required_actions, *answer.procedure_order),
        )
        forbidden_score = 0.0 if unsafe_recommendations else declared_forbidden_score
        rule_score, matched_rules, missing_rules = _coverage(
            expected.applicable_unit_ids,
            answer.applicable_rules,
        )
        procedure_score, matched_procedure = _ordered_coverage(
            expected.procedure_order,
            answer.procedure_order,
        )
        missing_procedure = _missing(expected.procedure_order, matched_procedure)
        term_score, matched_terms, missing_terms = _term_coverage(
            expected.terms,
            answer,
        )

        scores = {
            SAFETY_COMPONENT: required_score,
            FORBIDDEN_COMPONENT: forbidden_score,
            RULE_COMPONENT: rule_score,
            PROCEDURE_COMPONENT: procedure_score,
            TERMINOLOGY_COMPONENT: term_score,
        }
        evidence = {
            SAFETY_COMPONENT: matched_required,
            FORBIDDEN_COMPONENT: matched_forbidden,
            RULE_COMPONENT: matched_rules,
            PROCEDURE_COMPONENT: matched_procedure,
            TERMINOLOGY_COMPONENT: matched_terms,
        }
        notes = {
            SAFETY_COMPONENT: _missing_note(missing_required),
            FORBIDDEN_COMPONENT: _forbidden_note(
                missing_forbidden,
                unsafe_recommendations,
            ),
            RULE_COMPONENT: _missing_note(missing_rules),
            PROCEDURE_COMPONENT: _missing_note(missing_procedure),
            TERMINOLOGY_COMPONENT: _missing_note(missing_terms),
        }
        components = tuple(
            RewardComponent(
                component_id=component_id,
                weight=self.weights[component_id],
                score=scores[component_id],
                evidence=evidence[component_id],
                notes=notes[component_id],
                verifier_id=self.verifier_id,
            )
            for component_id in _COMPONENT_ORDER
        )

        gate_reasons: list[str] = []
        if self.safety_hard_gate and required_score < self.safety_threshold:
            gate_reasons.append(
                f"required-action coverage {required_score:.6g} is below "
                f"{self.safety_threshold:.6g}"
            )
        if self.safety_hard_gate and unsafe_recommendations:
            gate_reasons.append(
                "response positively recommends forbidden actions: "
                + "; ".join(unsafe_recommendations)
            )
        gate_multiplier = 0.0 if gate_reasons else 1.0
        diagnostics: dict[str, Any] = {
            "schema_validity": schema_validity,
            "source_grounding": rule_score,
            "unsafe_recommendation_detected": float(bool(unsafe_recommendations)),
            "safety_gate_applied": float(gate_multiplier == 0.0),
        }
        if gate_reasons:
            diagnostics["safety_gate_reasons"] = tuple(gate_reasons)

        return RewardLedger(
            task_id=task.task_id,
            response_id=resolved_response_id,
            components=components,
            gate_multiplier=gate_multiplier,
            diagnostics=diagnostics,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
        )

    def __call__(
        self,
        task: Task,
        response: StructuredAnswer | ModelResponse | Mapping[str, Any] | str,
        *,
        response_id: str | None = None,
    ) -> RewardLedger:
        return self.verify(task, response, response_id=response_id)

    def _invalid_ledger(
        self,
        task: Task,
        *,
        response_id: str,
        schema_validity: float,
    ) -> RewardLedger:
        components = tuple(
            RewardComponent(
                component_id=component_id,
                weight=self.weights[component_id],
                score=0.0,
                evidence=(),
                notes="response did not satisfy the structured answer contract",
                verifier_id=self.verifier_id,
            )
            for component_id in _COMPONENT_ORDER
        )
        return RewardLedger(
            task_id=task.task_id,
            response_id=response_id,
            components=components,
            gate_multiplier=0.0 if self.safety_hard_gate else 1.0,
            diagnostics={
                "schema_validity": schema_validity,
                "source_grounding": 0.0,
                "unsafe_recommendation_detected": 0.0,
                "safety_gate_applied": float(self.safety_hard_gate),
                "safety_gate_reasons": ("structured answer is invalid",),
            },
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
        )


CompositeVerifier = DeterministicVerifier


def verify_answer(
    task: Task,
    response: StructuredAnswer | ModelResponse | Mapping[str, Any] | str,
    *,
    weights: Mapping[str, float] | None = None,
    safety_hard_gate: bool = True,
    safety_threshold: float = 1.0,
    response_id: str | None = None,
) -> RewardLedger:
    """Convenience entry point for the default deterministic verifier."""

    verifier = DeterministicVerifier(
        weights,
        safety_hard_gate=safety_hard_gate,
        safety_threshold=safety_threshold,
    )
    return verifier.verify(task, response, response_id=response_id)


def _coerce_answer(
    task: Task,
    response: StructuredAnswer | ModelResponse | Mapping[str, Any] | str,
    *,
    response_id: str | None,
) -> tuple[StructuredAnswer | None, str, float]:
    if isinstance(response, ModelResponse):
        if response.task_id != task.task_id:
            raise ValueError(
                f"response task_id {response.task_id!r} does not match task {task.task_id!r}"
            )
        resolved_id = response_id or response.response_id
        if response.structured_answer is not None:
            return response.structured_answer, resolved_id, 1.0
        candidate: Any = response.raw_text
    else:
        resolved_id = response_id or f"{task.task_id}.response"
        candidate = response

    if isinstance(candidate, StructuredAnswer):
        return candidate, resolved_id, 1.0
    try:
        value = json.loads(candidate) if isinstance(candidate, str) else candidate
        return StructuredAnswer.model_validate(value), resolved_id, 1.0
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError):
        return None, resolved_id, 0.0


def _validated_weights(overrides: Mapping[str, float] | None) -> dict[str, float]:
    selected = dict(DEFAULT_WEIGHTS)
    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_WEIGHTS))
        if unknown:
            raise ValueError("unknown reward components: " + ", ".join(unknown))
        selected.update(overrides)
    if any(not math.isfinite(value) or value < 0 for value in selected.values()):
        raise ValueError("reward weights must be finite and non-negative")
    total = sum(selected.values())
    if total <= 0:
        raise ValueError("at least one reward weight must be greater than zero")
    return {component_id: selected[component_id] / total for component_id in _COMPONENT_ORDER}


def normalize_text(value: str) -> str:
    """Normalize case, compatibility characters, punctuation, and whitespace."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = _NON_WORD_RE.sub(" ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def _coverage(
    expected: Sequence[str],
    actual: Sequence[str],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    expected_values = _unique(expected)
    if not expected_values:
        return 1.0, (), ()
    actual_keys = {normalize_text(value) for value in actual}
    matched = tuple(value for value in expected_values if normalize_text(value) in actual_keys)
    missing = tuple(value for value in expected_values if normalize_text(value) not in actual_keys)
    return len(matched) / len(expected_values), matched, missing


def _intersection(expected: Sequence[str], actual: Sequence[str]) -> tuple[str, ...]:
    actual_keys = {normalize_text(value) for value in actual}
    return tuple(value for value in _unique(expected) if normalize_text(value) in actual_keys)


def _ordered_coverage(
    expected: Sequence[str],
    actual: Sequence[str],
) -> tuple[float, tuple[str, ...]]:
    expected_values = _unique(expected)
    if not expected_values:
        return 1.0, ()
    expected_keys = tuple(normalize_text(value) for value in expected_values)
    actual_keys = tuple(normalize_text(value) for value in actual)
    matched_indexes = _longest_common_subsequence_indexes(expected_keys, actual_keys)
    matched = tuple(expected_values[index] for index in matched_indexes)
    return len(matched) / len(expected_values), matched


def _longest_common_subsequence_indexes(
    expected: Sequence[str],
    actual: Sequence[str],
) -> tuple[int, ...]:
    rows = len(expected) + 1
    columns = len(actual) + 1
    lengths = [[0] * columns for _ in range(rows)]
    for left in range(1, rows):
        for right in range(1, columns):
            if expected[left - 1] == actual[right - 1]:
                lengths[left][right] = lengths[left - 1][right - 1] + 1
            else:
                lengths[left][right] = max(lengths[left - 1][right], lengths[left][right - 1])

    indexes: list[int] = []
    left, right = len(expected), len(actual)
    while left and right:
        if expected[left - 1] == actual[right - 1]:
            indexes.append(left - 1)
            left -= 1
            right -= 1
        elif lengths[left - 1][right] >= lengths[left][right - 1]:
            left -= 1
        else:
            right -= 1
    indexes.reverse()
    return tuple(indexes)


def _term_coverage(
    expected: Sequence[str],
    answer: StructuredAnswer,
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    expected_values = _unique(expected)
    if not expected_values:
        return 1.0, (), ()
    # Terminology is graded from the user-visible prose, not from hidden answer
    # contract fields that may have been copied mechanically.
    searchable = normalize_text(answer.final_answer)
    padded = f" {searchable} "
    matched = tuple(
        term for term in expected_values if f" {normalize_text(term)} " in padded
    )
    missing = tuple(term for term in expected_values if term not in matched)
    return len(matched) / len(expected_values), matched, missing


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(str(value))
    return tuple(result)


def _missing(expected: Sequence[str], matched: Sequence[str]) -> tuple[str, ...]:
    matched_keys = {normalize_text(value) for value in matched}
    return tuple(value for value in _unique(expected) if normalize_text(value) not in matched_keys)


def _missing_note(missing: Sequence[str]) -> str | None:
    if not missing:
        return None
    return "missing: " + "; ".join(missing)


def _forbidden_note(
    missing: Sequence[str],
    unsafe_recommendations: Sequence[str],
) -> str | None:
    notes: list[str] = []
    if missing:
        notes.append("not identified as forbidden: " + "; ".join(missing))
    if unsafe_recommendations:
        notes.append("positively recommended: " + "; ".join(unsafe_recommendations))
    return " | ".join(notes) or None


__all__ = [
    "CompositeVerifier",
    "DEFAULT_WEIGHTS",
    "DeterministicVerifier",
    "FORBIDDEN_COMPONENT",
    "PROCEDURE_COMPONENT",
    "RULE_COMPONENT",
    "SAFETY_COMPONENT",
    "TERMINOLOGY_COMPONENT",
    "normalize_text",
    "verify_answer",
]
