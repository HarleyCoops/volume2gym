import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from deploy.server import (
    Handler,
    _available_volumes,
    _compile,
    _compile_named_volume,
    _load_volume_descriptor,
    _reference_eval,
    _reference_eval_named_volume,
)
from scripts.smoke_hosted_deployment import run_smoke

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


@pytest.fixture
def hosted_server(monkeypatch):
    monkeypatch.setenv("SOURCE2AGENT_ALLOW_CUSTOM_INPUT", "0")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "test-revision")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_hosted_named_volume_smoke_contract(hosted_server):
    evidence = run_smoke(hosted_server, expected_revision="test-revision")

    assert evidence["provider"] == "railway"
    assert evidence["observed_revision"] == "test-revision"
    assert evidence["checks"]["compile"]["task_count"] == 2742
    assert evidence["checks"]["reference_eval"]["task_count"] == 276
    assert evidence["neural_model"] is False


def test_hosted_mode_rejects_custom_inputs(hosted_server):
    request = Request(
        f"{hosted_server}/compile",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=5)  # noqa: S310 - local test server

    assert exc_info.value.code == 403
    payload = json.loads(exc_info.value.read())
    assert payload["named_volume_endpoints_available"] is True


def test_named_endpoint_rejects_oversized_body(hosted_server):
    request = Request(
        f"{hosted_server}/volumes/railroad-1959-v0/compile",
        data=b"x" * (256 * 1024 + 1),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=5)  # noqa: S310 - local test server

    assert exc_info.value.code == 413
    assert "request body exceeds" in json.loads(exc_info.value.read())["error"]


def test_custom_unit_limit(monkeypatch):
    monkeypatch.setenv("SOURCE2AGENT_MAX_CUSTOM_UNITS", "2")
    with pytest.raises(ValueError, match="configured limit of 2"):
        _compile(
            {
                "volume_id": "too-large",
                "units": load_units(),
                "seed": 7,
            }
        )
