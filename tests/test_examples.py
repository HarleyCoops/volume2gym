from pathlib import Path

from volume2gym.environment import VolumeGym
from volume2gym.exporters import reference_answer
from volume2gym.pipeline import compile_build, validate_build

ROOT = Path(__file__).resolve().parents[1]


def test_lantern_ledger_closes_the_generic_volume_to_gym_loop(tmp_path) -> None:
    result = compile_build(
        volume_id="lantern-ledger-demo",
        output_dir=tmp_path / "build",
        canonical_units_path=ROOT / "examples" / "lantern_ledger" / "units.json",
        seed=7,
    )

    summary = validate_build(result.root)
    assert summary["knowledge_unit_count"] == 3
    assert summary["task_count"] == 18
    assert summary["split_counts"] == {"train": 6, "dev": 6, "test": 6}

    task = result.tasks[0]
    gym = VolumeGym([task])
    observation, info = gym.reset(options={"task_id": task.task_id})
    _, reward, terminated, truncated, step_info = gym.step(reference_answer(task))

    assert observation == task.prompt
    assert info["task_id"] == task.task_id
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert step_info["reward_ledger"]["total_score"] == 1.0
