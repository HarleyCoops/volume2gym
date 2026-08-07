"""Small dependency-light Source2Agent artifact service.

This service deliberately exposes the checked compiler/reference path rather than
pretending that a symbolic answer key is a trained model. Hosted neural inference
is a separate Prime Lab integration and must publish its own evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from volume2gym.pipeline import compile_build, inspect_build, validate_build
from volume2gym.trainers import SymbolicTrainer, evaluate_policy
from volume2gym.pipeline import load_build


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _compile(payload: dict[str, Any]) -> dict[str, Any]:
    volume_id = str(payload.get("volume_id", "source2agent-request"))
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("payload.units must be a non-empty list")

    with tempfile.TemporaryDirectory(prefix="source2agent-") as work:
        root = Path(work)
        units_path = root / "units.json"
        build_path = root / "build"
        units_path.write_text(
            json.dumps(units, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compile_build(
            volume_id=volume_id,
            output_dir=build_path,
            canonical_units_path=units_path,
            seed=int(payload.get("seed", 7)),
        )
        validated = validate_build(build_path)
        inspected = inspect_build(build_path)
        return {
            "product": "Source2Agent",
            "engine": "volume2gym",
            "mode": "local-compiler",
            "volume_id": volume_id,
            "validated": _jsonable(validated),
            "inspection": _jsonable(inspected),
        }


def _reference_eval(payload: dict[str, Any]) -> dict[str, Any]:
    volume_id = str(payload.get("volume_id", "source2agent-request"))
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("payload.units must be a non-empty list")

    with tempfile.TemporaryDirectory(prefix="source2agent-eval-") as work:
        root = Path(work)
        units_path = root / "units.json"
        build_path = root / "build"
        units_path.write_text(
            json.dumps(units, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compile_build(
            volume_id=volume_id,
            output_dir=build_path,
            canonical_units_path=units_path,
            seed=int(payload.get("seed", 7)),
        )
        tasks = load_build(build_path).tasks
        policy = SymbolicTrainer().train(tasks)
        records = evaluate_policy(policy, tasks)
        scores = [
            float(record.reward_ledger.total_score or 0.0)
            for record in records
        ]
        return {
            "product": "Source2Agent",
            "engine": "volume2gym",
            "mode": "symbolic-reference",
            "volume_id": volume_id,
            "model_id": policy.model_id,
            "task_count": len(records),
            "mean_total_score": sum(scores) / len(scores),
            "neural_model": False,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "Source2Agent/0.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "healthy", "product": "Source2Agent"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/compile", "/reference-eval"}:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            result = (
                _compile(payload)
                if self.path == "/compile"
                else _reference_eval(payload)
            )
            self._send(200, result)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Source2Agent listening on :{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
