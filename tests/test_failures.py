from volume2gym.failures import build_curriculum, cluster_failures, curriculum_records
from volume2gym.models import AnswerKey, StructuredAnswer, Task, TaskFamily
from volume2gym.verifier import SAFETY_COMPONENT, TERMINOLOGY_COMPONENT, DeterministicVerifier


def make_task(task_id):
    return Task(
        task_id=task_id,
        prompt="Respond safely.",
        task_family=TaskFamily("standard_operation"),
        knowledge_unit_ids=("unit-safety",),
        answer_key=AnswerKey(
            applicable_unit_ids=("unit-safety",),
            required_actions=("protect", "communicate"),
            forbidden_actions=("proceed unsafely",),
            procedure_order=("protect", "communicate"),
            terms=("flagman",),
        ),
    )


def make_answer(*, include_all_actions=True, include_term=True):
    return StructuredAnswer(
        applicable_rules=("unit-safety",),
        required_actions=("protect", "communicate") if include_all_actions else ("protect",),
        forbidden_actions=("proceed unsafely",),
        procedure_order=("protect", "communicate"),
        final_answer="The flagman protects first." if include_term else "Protection comes first.",
    )


def test_component_failures_cluster_and_safety_ranks_first():
    task_a = make_task("task-a")
    task_b = make_task("task-b")
    verifier = DeterministicVerifier()
    ledgers = [
        verifier.verify(task_a, make_answer(include_all_actions=False)),
        verifier.verify(task_b, make_answer(include_term=False)),
    ]

    clusters = cluster_failures(ledgers, tasks=[task_a, task_b])

    assert clusters[0].component == SAFETY_COMPONENT
    assert clusters[0].severity.value == "critical"
    assert clusters[0].knowledge_unit_ids == ("unit-safety",)
    assert {cluster.component for cluster in clusters} == {
        SAFETY_COMPONENT,
        TERMINOLOGY_COMPONENT,
    }


def test_curriculum_retains_source_tasks_actions_and_unsafe_examples():
    task = make_task("task-a")
    ledger = DeterministicVerifier().verify(
        task,
        make_answer(include_all_actions=False),
    )
    clusters = cluster_failures([ledger], tasks=[task])
    targets = build_curriculum(clusters, [task], limit=1)

    assert targets[0].target_component == SAFETY_COMPONENT
    assert targets[0].source_task_ids == ("task-a",)
    assert targets[0].expected_required_actions == ("protect", "communicate")
    assert targets[0].unsafe_counterexamples == ("proceed unsafely",)
    assert curriculum_records(clusters, [task], limit=1)[0]["priority"] == 1


def test_threshold_and_minimum_count_control_cluster_creation():
    task = make_task("task-a")
    ledger = DeterministicVerifier().verify(task, make_answer(include_term=False))

    assert cluster_failures([ledger], tasks=[task], minimum_count=2) == []
    assert cluster_failures([ledger], tasks=[task], threshold=0.0) == []
