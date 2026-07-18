from __future__ import annotations

import pytest

from volume2gym.compiler import CORE_TASK_FAMILIES, TaskTemplate, TemplateCompiler, compile_units
from volume2gym.models import Citation, KnowledgeKind, KnowledgeUnit, TaskFamily


def _unit(unit_id: str = "unit-alpha", *, family: str = "operations") -> KnowledgeUnit:
    citation = Citation(
        document_id="manual-a",
        span_id=f"span-{unit_id}",
        page_number=12,
        section="Operations",
        quote="Stop before crossing the marked boundary.",
    )
    return KnowledgeUnit(
        unit_id=unit_id,
        kind=KnowledgeKind.RULE,
        title="Marked boundary procedure",
        text="Stop before crossing the marked boundary.",
        family=family,
        section="Operations",
        conditions=("A marked boundary is ahead.",),
        required_actions=("Stop before the boundary.",),
        forbidden_actions=("Cross without authorization.",),
        procedure_steps=("Observe the marker.", "Stop.", "Obtain authorization."),
        exceptions=("Proceed only when an authorized exception applies.",),
        terms=("marked boundary", "authorization"),
        citations=(citation,),
    )


def test_compiles_every_unit_through_six_source_grounded_families() -> None:
    unit = _unit()

    tasks = compile_units([unit])

    assert len(tasks) == 6
    assert tuple(task.task_family for task in tasks) == CORE_TASK_FAMILIES
    assert len({task.task_id for task in tasks}) == 6
    for task in tasks:
        assert task.knowledge_unit_ids == (unit.unit_id,)
        assert task.rule_family == unit.family
        assert task.answer_key.applicable_unit_ids == (unit.unit_id,)
        assert task.answer_key.required_actions == unit.required_actions
        assert task.answer_key.forbidden_actions == unit.forbidden_actions
        assert task.answer_key.procedure_order == unit.procedure_steps
        assert task.answer_key.citations == unit.citations
        assert unit.unit_id in task.prompt
        assert unit.title in task.prompt
        assert unit.text in task.prompt
        assert task.generator == "volume2gym.template.v1"


def test_compilation_is_stable_across_runs_and_input_order() -> None:
    units = [_unit("zeta", family="z"), _unit("alpha", family="a")]

    forward = TemplateCompiler().compile(units)
    reverse = TemplateCompiler().compile(reversed(units))

    assert [(task.task_id, task.prompt) for task in forward] == [
        (task.task_id, task.prompt) for task in reverse
    ]
    assert forward[0].knowledge_unit_ids == ("alpha",)


def test_duplicate_unit_ids_fail_instead_of_silently_overwriting() -> None:
    with pytest.raises(ValueError, match="duplicate knowledge unit ids"):
        TemplateCompiler().compile([_unit("same"), _unit("same")])


def test_custom_templates_must_retain_all_six_families() -> None:
    partial = [
        TaskTemplate(
            task_family=TaskFamily("standard_operation"),
            instruction="Apply the source.",
            tags=("custom",),
        )
    ]

    with pytest.raises(ValueError, match="missing core task families"):
        TemplateCompiler(partial)
