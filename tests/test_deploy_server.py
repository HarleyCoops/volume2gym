import json
from pathlib import Path

from deploy.server import (
    _available_volumes,
    _compile,
    _compile_named_volume,
    _load_volume_descriptor,
    _reference_eval,
    _reference_eval_named_volume,
)

FIXTURE = Path(__file__).parents[1] / "examples" / "lantern_ledger" / "units.json"


def load_units():
    return json.loads(FIXTURE.read_text())


def test_deployed_compile_contract():
    result = _compile(
        {
            "volume_id": "lantern-ledger-deploy-test",
            "units": load_units(),
            "seed": 7,
        }
    )

    assert result["product"] == "Source2Agent"
    assert result["engine"] == "volume2gym"
    assert result["validated"]["valid"] is True
    assert result["validated"]["task_count"] == 18
    assert result["validated"]["split_counts"] == {
        "train": 6,
        "dev": 6,
        "test": 6,
    }
    assert result["validated"]["artifact_count"] == 5
    assert result["inspection"]["task_family_counts"]


def test_deployed_reference_eval_contract():
    result = _reference_eval(
        {
            "volume_id": "lantern-ledger-deploy-test",
            "units": load_units(),
            "seed": 7,
        }
    )

    assert result["product"] == "Source2Agent"
    assert result["mode"] == "symbolic-reference"
    assert result["task_count"] == 18
    assert result["mean_total_score"] == 1.0
    assert result["neural_model"] is False


def write_named_fixture(tmp_path):
    volume_dir = tmp_path / "named-fixture"
    volume_dir.mkdir()
    (volume_dir / "rules.json").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "rule_id": "A",
                        "text": "Keep the lantern lit.",
                        "page_number": 4,
                    },
                    {
                        "rule_id": "B",
                        "text": "Record every crossing.",
                        "page_number": 4,
                    },
                    {
                        "rule_id": "C",
                        "text": "Stop when the ledger is missing.",
                        "page_number": 4,
                    },
                ]
            }
        )
    )
    descriptor = {
        "volume_id": "named-fixture",
        "document": {"document_id": "fixture", "source_revision": "abc123"},
        "corpus": {"rules_path": "rules.json"},
        "build": {"seed": 7, "group_by": "knowledge_unit"},
    }
    (volume_dir / "volume.json").write_text(json.dumps(descriptor))
    return descriptor


def test_named_volume_compile_and_reference_endpoints(tmp_path):
    descriptor = write_named_fixture(tmp_path)

    assert _available_volumes(volume_root=tmp_path) == [descriptor]
    assert _load_volume_descriptor("named-fixture", volume_root=tmp_path) == descriptor

    compiled = _compile_named_volume("named-fixture", volume_root=tmp_path)
    assert compiled["mode"] == "named-volume-compiler"
    assert compiled["validated"]["valid"] is True
    assert compiled["validated"]["knowledge_unit_count"] == 3
    assert compiled["validated"]["task_count"] == 18

    evaluated = _reference_eval_named_volume("named-fixture", volume_root=tmp_path)
    assert evaluated["mode"] == "symbolic-reference"
    assert evaluated["task_count"] == 6
    assert evaluated["mean_total_score"] == 1.0
    assert evaluated["neural_model"] is False
