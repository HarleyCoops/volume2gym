from pathlib import Path

from volume2gym.integrations.huggingface import load_huggingface_fixture
from volume2gym.models import Split

ROOT = Path(__file__).parents[1]


def test_published_fixture_imports_without_rule_specific_code():
    fixture = load_huggingface_fixture(ROOT / "artifacts/huggingface/rule99_dataset")

    assert len(fixture.knowledge_units) == 1
    assert len(fixture.tasks) == 7
    assert sum(task.split == Split.TRAIN for task in fixture.tasks) == 6
    assert sum(task.split == Split.TEST for task in fixture.tasks) == 1
    assert all(task.citations for task in fixture.tasks)
