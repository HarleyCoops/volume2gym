import json
from pathlib import Path

from deploy.server import _compile, _reference_eval


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
    assert result["task_count"] == 18
    assert result["split_counts"] == {"train": 6, "dev": 6, "test": 6}
    assert result["artifact_count"] == 5


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
