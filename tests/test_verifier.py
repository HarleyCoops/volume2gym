import pytest

from volume2gym.models import AnswerKey, ModelResponse, StructuredAnswer, Task, TaskFamily
from volume2gym.verifier import (
    DEFAULT_WEIGHTS,
    SAFETY_COMPONENT,
    TERMINOLOGY_COMPONENT,
    DeterministicVerifier,
)


def make_task(task_id="task-99"):
    return Task(
        task_id=task_id,
        prompt="What is the safe action?",
        task_family=TaskFamily("standard_operation"),
        knowledge_unit_ids=("unit-99",),
        answer_key=AnswerKey(
            applicable_unit_ids=("unit-99",),
            required_actions=("protect the rear", "place torpedoes"),
            forbidden_actions=("leave the train unprotected",),
            procedure_order=("recognize exposure", "establish protection"),
            terms=("flagman", "torpedoes"),
        ),
    )


def perfect_answer():
    return StructuredAnswer(
        applicable_rules=("unit-99",),
        situation_type="stopped train",
        required_actions=("Protect the rear!", "Place torpedoes."),
        forbidden_actions=("Leave the train unprotected",),
        procedure_order=("Recognize exposure", "Establish protection"),
        final_answer="The flagman must protect the rear and place torpedoes.",
    )


def component_map(ledger):
    return {component.component_id: component for component in ledger.components}


def test_perfect_structured_answer_scores_one():
    ledger = DeterministicVerifier().verify(make_task(), perfect_answer())

    assert ledger.total_score == pytest.approx(1.0)
    assert ledger.ungated_total_score == pytest.approx(1.0)
    assert ledger.gate_multiplier == 1.0
    assert ledger.diagnostics["unsafe_recommendation_detected"] == 0.0
    assert set(component_map(ledger)) == set(DEFAULT_WEIGHTS)


def test_missing_required_action_triggers_safety_gate_but_preserves_raw_score():
    answer = perfect_answer().model_copy(update={"required_actions": ("protect the rear",)})
    ledger = DeterministicVerifier().verify(make_task(), answer)
    components = component_map(ledger)

    assert components[SAFETY_COMPONENT].score == pytest.approx(0.5)
    assert ledger.ungated_total_score > 0.0
    assert ledger.gate_multiplier == 0.0
    assert ledger.total_score == 0.0
    assert ledger.diagnostics["safety_gate_applied"] == 1.0


def test_positively_recommending_forbidden_action_triggers_gate():
    answer = perfect_answer().model_copy(
        update={
            "required_actions": (
                "protect the rear",
                "place torpedoes",
                "leave the train unprotected",
            )
        }
    )
    ledger = DeterministicVerifier().verify(make_task(), answer)

    assert ledger.total_score == 0.0
    assert ledger.diagnostics["unsafe_recommendation_detected"] == 1.0


def test_custom_weights_are_normalized_and_configurable():
    weights = {component: 0.0 for component in DEFAULT_WEIGHTS}
    weights[TERMINOLOGY_COMPONENT] = 7.0
    verifier = DeterministicVerifier(weights, safety_hard_gate=False)
    answer = perfect_answer().model_copy(update={"final_answer": "No domain terms here."})
    ledger = verifier.verify(make_task(), answer)

    assert verifier.weights[TERMINOLOGY_COMPONENT] == 1.0
    assert ledger.total_score == 0.0


def test_invalid_raw_response_returns_schema_failure_instead_of_crashing():
    response = ModelResponse(
        response_id="response-1",
        task_id="task-99",
        raw_text="not-json",
    )
    ledger = DeterministicVerifier().verify(make_task(), response)

    assert ledger.response_id == "response-1"
    assert ledger.total_score == 0.0
    assert ledger.diagnostics["schema_validity"] == 0.0


def test_response_cannot_be_scored_against_another_task():
    response = ModelResponse(
        response_id="response-1",
        task_id="different-task",
        raw_text="{}",
    )
    with pytest.raises(ValueError, match="does not match"):
        DeterministicVerifier().verify(make_task(), response)


def test_weight_and_threshold_validation():
    with pytest.raises(ValueError, match="unknown reward components"):
        DeterministicVerifier({"invented": 1.0})
    with pytest.raises(ValueError, match="between 0 and 1"):
        DeterministicVerifier(safety_threshold=1.1)
