"""Provider-neutral compilation of knowledge units into trainable tasks.

The compiler in this module is deliberately deterministic and contains no model
client.  A domain profile may enrich :class:`KnowledgeUnit` objects before they
arrive here, but the same templates work for rules, procedures, definitions,
standards, and other structured source units.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256

from .models import AnswerContract, AnswerKey, KnowledgeUnit, Task, TaskFamily

TEMPLATE_VERSION = "volume2gym.template.v1"


def _family(value: str) -> TaskFamily:
    """Resolve enum values without depending on enum member naming style."""

    return TaskFamily(value)


CORE_TASK_FAMILIES: tuple[TaskFamily, ...] = (
    _family("standard_operation"),
    _family("edge_case"),
    _family("conflict_resolution"),
    _family("exception_handling"),
    _family("violation_check"),
    _family("adversarial_distractor"),
)


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    """A deterministic instruction template for one task family."""

    task_family: TaskFamily
    instruction: str
    tags: tuple[str, ...]


DEFAULT_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate(
        _family("standard_operation"),
        (
            "Describe the correct ordinary application of this source unit. "
            "State the applicable conditions, required actions, forbidden "
            "actions, and procedure order without inventing requirements."
        ),
        ("standard", "application"),
    ),
    TaskTemplate(
        _family("edge_case"),
        (
            "Analyze a boundary or ambiguous case for this source unit. "
            "Explain which stated conditions still control and identify what "
            "additional facts would be needed if the source is not decisive."
        ),
        ("edge-case", "conditions"),
    ),
    TaskTemplate(
        _family("conflict_resolution"),
        (
            "Resolve any apparent tension among this unit's conditions, "
            "requirements, prohibitions, procedure, and exceptions. Preserve "
            "every constraint that can be satisfied together."
        ),
        ("conflict", "constraints"),
    ),
    TaskTemplate(
        _family("exception_handling"),
        (
            "Determine whether an exception changes the normal application of "
            "this source unit. Name the exception and its triggering conditions, "
            "or state explicitly that the source records no exception."
        ),
        ("exception", "conditions"),
    ),
    TaskTemplate(
        _family("violation_check"),
        (
            "Evaluate a proposed response for violations of this source unit. "
            "Identify omitted required actions, included forbidden actions, and "
            "out-of-order procedure steps, then give a compliant correction."
        ),
        ("violation", "verification"),
    ),
    TaskTemplate(
        _family("adversarial_distractor"),
        (
            "Reject a plausible but unsupported instruction that conflicts with "
            "this source unit. Base the response only on the cited source and "
            "separate sourced requirements from assumptions."
        ),
        ("adversarial", "grounding"),
    ),
)


class TemplateCompiler:
    """Compile every knowledge unit through the same six task families.

    The output is stable across input order and runs. Custom templates can
    replace instructions, but all six core families remain present so builds do
    not silently lose coverage.
    """

    def __init__(
        self,
        templates: Iterable[TaskTemplate] | None = None,
        *,
        answer_contract: AnswerContract | None = None,
        generator: str = TEMPLATE_VERSION,
    ) -> None:
        selected = tuple(templates or DEFAULT_TEMPLATES)
        by_family = {template.task_family: template for template in selected}
        missing = tuple(family.value for family in CORE_TASK_FAMILIES if family not in by_family)
        if missing:
            raise ValueError(f"templates missing core task families: {', '.join(missing)}")
        if len(by_family) != len(selected):
            raise ValueError("templates must contain at most one entry per task family")

        self._templates: Mapping[TaskFamily, TaskTemplate] = by_family
        self._answer_contract = answer_contract or AnswerContract()
        self.generator = generator

    @property
    def templates(self) -> tuple[TaskTemplate, ...]:
        return tuple(self._templates[family] for family in CORE_TASK_FAMILIES)

    def compile(self, units: Iterable[KnowledgeUnit]) -> list[Task]:
        """Compile units into six tasks each, sorted by stable unit identifier."""

        ordered = sorted(tuple(units), key=lambda unit: unit.unit_id)
        counts = Counter(unit.unit_id for unit in ordered)
        duplicates = sorted(item for item, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate knowledge unit ids: {', '.join(duplicates)}")

        return [task for unit in ordered for task in self.compile_unit(unit)]

    def compile_unit(self, unit: KnowledgeUnit) -> list[Task]:
        """Compile one source-grounded task for each core task family."""

        return [self._compile_task(unit, self._templates[family]) for family in CORE_TASK_FAMILIES]

    def _compile_task(self, unit: KnowledgeUnit, template: TaskTemplate) -> Task:
        rule_family = unit.family or unit.section or unit.unit_id
        metadata = {
            "template_version": TEMPLATE_VERSION,
            "primary_unit_id": unit.unit_id,
            "source_kind": _enum_value(unit.kind),
        }
        answer_key = AnswerKey(
            applicable_unit_ids=(unit.unit_id,),
            required_actions=tuple(unit.required_actions),
            forbidden_actions=tuple(unit.forbidden_actions),
            procedure_order=tuple(unit.procedure_steps),
            terms=tuple(unit.terms),
            reference_answer=_reference_answer(unit),
            citations=tuple(unit.citations),
        )

        return Task(
            task_id=_task_id(unit.unit_id, template.task_family, self.generator),
            prompt=_render_prompt(unit, template),
            task_family=template.task_family,
            knowledge_unit_ids=(unit.unit_id,),
            answer_contract=self._answer_contract,
            answer_key=answer_key,
            citations=tuple(unit.citations),
            rule_family=rule_family,
            tags=("template-compiled", *template.tags),
            generator=self.generator,
            metadata=metadata,
        )


def compile_units(
    units: Iterable[KnowledgeUnit],
    *,
    compiler: TemplateCompiler | None = None,
) -> list[Task]:
    """Convenience entry point for the default deterministic compiler."""

    return (compiler or TemplateCompiler()).compile(units)


def _task_id(unit_id: str, family: TaskFamily, generator: str) -> str:
    digest = sha256(f"{generator}\0{unit_id}\0{family.value}".encode()).hexdigest()[:20]
    return f"v2g-{digest}"


def _render_prompt(unit: KnowledgeUnit, template: TaskTemplate) -> str:
    lines = [
        f"Task family: {template.task_family.value}",
        f"Source unit: {unit.unit_id}",
        f"Title: {unit.title}",
    ]
    if unit.section:
        lines.append(f"Section: {unit.section}")
    lines.extend(("", "Source text:", unit.text.strip(), "", "Assignment:", template.instruction))
    return "\n".join(lines).strip()


def _reference_answer(unit: KnowledgeUnit) -> str:
    """Build a source-faithful reference without adding domain assertions."""

    sections = [unit.text.strip()]
    structured = (
        ("Conditions", unit.conditions),
        ("Required actions", unit.required_actions),
        ("Forbidden actions", unit.forbidden_actions),
        ("Procedure", unit.procedure_steps),
        ("Exceptions", unit.exceptions),
        ("Terms", unit.terms),
    )
    for heading, values in structured:
        if values:
            sections.append(f"{heading}: " + "; ".join(str(value) for value in values))
    return "\n".join(sections)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


__all__ = [
    "CORE_TASK_FAMILIES",
    "DEFAULT_TEMPLATES",
    "TEMPLATE_VERSION",
    "TaskTemplate",
    "TemplateCompiler",
    "compile_units",
]
