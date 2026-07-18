from volume2gym.models import AnswerKey, Task, TaskFamily
from volume2gym.trainers import SymbolicTrainer, evaluate_policy


def test_symbolic_reference_closes_the_training_and_evaluation_loop():
    task = Task(
        task_id="task-1",
        prompt="Act safely.",
        task_family=TaskFamily.STANDARD_OPERATION,
        knowledge_unit_ids=("unit-1",),
        answer_key=AnswerKey(
            applicable_unit_ids=("unit-1",),
            required_actions=("stop",),
            forbidden_actions=("continue",),
            procedure_order=("recognize", "stop"),
            terms=("stop",),
            reference_answer="Stop.",
        ),
    )
    policy = SymbolicTrainer().train([task])
    records = evaluate_policy(policy, [task])

    assert records[0].reward_ledger.total_score == 1.0
    assert records[0].response.model_id == "volume2gym-symbolic-reference"
