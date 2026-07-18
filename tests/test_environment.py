import pytest

from volume2gym.environment import VolumeGym
from volume2gym.models import AnswerKey, StructuredAnswer, Task, TaskFamily


def make_task(task_id):
    return Task(
        task_id=task_id,
        prompt=f"Prompt for {task_id}",
        task_family=TaskFamily("standard_operation"),
        knowledge_unit_ids=(f"unit-{task_id}",),
        answer_key=AnswerKey(
            applicable_unit_ids=(f"unit-{task_id}",),
            required_actions=("stop",),
            forbidden_actions=("continue",),
            procedure_order=("recognize", "stop"),
            terms=("stop",),
        ),
    )


def answer_for(task_id):
    return StructuredAnswer(
        applicable_rules=(f"unit-{task_id}",),
        required_actions=("stop",),
        forbidden_actions=("continue",),
        procedure_order=("recognize", "stop"),
        final_answer="Stop.",
    )


def test_single_turn_reset_and_step_return_gym_shapes():
    gym = VolumeGym([make_task("a"), make_task("b")])
    observation, info = gym.reset(options={"task_id": "b"})

    assert observation == "Prompt for b"
    assert info["task_id"] == "b"

    next_observation, reward, terminated, truncated, step_info = gym.step(answer_for("b"))
    assert next_observation == "Task complete"
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False
    assert step_info["reward_ledger"]["task_id"] == "b"

    with pytest.raises(RuntimeError, match="terminated"):
        gym.step(answer_for("b"))


def test_private_seeded_sampling_is_reproducible():
    tasks = [make_task("a"), make_task("b"), make_task("c")]
    left = VolumeGym(tasks)
    right = VolumeGym(tasks)

    assert left.reset(seed=81)[1]["task_id"] == right.reset(seed=81)[1]["task_id"]


def test_step_requires_reset_and_unknown_task_is_rejected():
    gym = VolumeGym([make_task("a")])
    with pytest.raises(RuntimeError, match="reset"):
        gym.step(answer_for("a"))
    with pytest.raises(KeyError, match="unknown task_id"):
        gym.reset(options={"task_id": "missing"})


def test_invalid_structured_action_is_a_zero_reward_episode():
    gym = VolumeGym([make_task("a")])
    gym.reset()
    _, reward, terminated, _, info = gym.step("not-json")

    assert reward == 0.0
    assert terminated is True
    assert info["reward_ledger"]["diagnostics"]["schema_validity"] == 0.0


def test_environment_validates_task_collection():
    with pytest.raises(ValueError, match="at least one"):
        VolumeGym([])
    duplicate = make_task("same")
    with pytest.raises(ValueError, match="duplicate task ids"):
        VolumeGym([duplicate, duplicate])
