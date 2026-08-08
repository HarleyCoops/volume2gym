"""Small dependency-light Source2Agent artifact service.

This service deliberately exposes the checked compiler/reference path rather than
pretending that a symbolic answer key is a trained model. Hosted neural inference
is a separate Prime Lab integration and must publish its own evidence.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from volume2gym.models import Split
from volume2gym.pipeline import compile_build, inspect_build, load_build, validate_build
from volume2gym.trainers import SymbolicTrainer, evaluate_policy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME_ROOT = REPOSITORY_ROOT / "volumes"
DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_MAX_CUSTOM_UNITS = 100
_COMPUTE_SLOT = threading.BoundedSemaphore(value=1)
_NAMED_RESULT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class ServiceBusyError(RuntimeError):
    """Raised when the bounded public compute slot is already occupied."""


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _custom_inputs_enabled() -> bool:
    return os.environ.get("SOURCE2AGENT_ALLOW_CUSTOM_INPUT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _validate_custom_units(payload: dict[str, Any]) -> list[Any]:
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("payload.units must be a non-empty list")
    maximum = _positive_int_env("SOURCE2AGENT_MAX_CUSTOM_UNITS", DEFAULT_MAX_CUSTOM_UNITS)
    if len(units) > maximum:
        raise ValueError(f"payload.units exceeds the configured limit of {maximum}")
    return units


def _run_bounded(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if not _COMPUTE_SLOT.acquire(blocking=False):
        raise ServiceBusyError("another compile or reference evaluation is in progress")
    try:
        return operation()
    finally:
        _COMPUTE_SLOT.release()


def _cached_named_result(
    volume_id: str,
    operation: str,
    compute: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    key = (volume_id, operation)
    with _CACHE_LOCK:
        cached = _NAMED_RESULT_CACHE.get(key)
    if cached is not None:
        return cached
    result = _run_bounded(compute)
    with _CACHE_LOCK:
        _NAMED_RESULT_CACHE[key] = result
    return result


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
    units = _validate_custom_units(payload)

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
    units = _validate_custom_units(payload)

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


def _volume_root(override: str | Path | None = None) -> Path:
    configured = override or os.environ.get("SOURCE2AGENT_VOLUME_ROOT") or DEFAULT_VOLUME_ROOT
    return Path(configured).resolve()


def _safe_volume_dir(volume_id: str, *, volume_root: str | Path | None = None) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not volume_id or any(character not in allowed for character in volume_id):
        raise ValueError(
            "volume_id may contain only lowercase letters, digits, hyphens, and underscores"
        )
    root = _volume_root(volume_root)
    candidate = (root / volume_id).resolve()
    if candidate.parent != root:
        raise ValueError("volume_id resolves outside the volume root")
    return candidate


def _load_volume_descriptor(
    volume_id: str,
    *,
    volume_root: str | Path | None = None,
) -> dict[str, Any]:
    descriptor_path = _safe_volume_dir(volume_id, volume_root=volume_root) / "volume.json"
    if not descriptor_path.is_file():
        raise FileNotFoundError(f"unknown volume: {volume_id}")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict) or descriptor.get("volume_id") != volume_id:
        raise ValueError(f"invalid descriptor for volume {volume_id}")
    return descriptor


def _available_volumes(*, volume_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _volume_root(volume_root)
    if not root.is_dir():
        return []
    descriptors: list[dict[str, Any]] = []
    for descriptor_path in sorted(root.glob("*/volume.json")):
        descriptors.append(
            _load_volume_descriptor(descriptor_path.parent.name, volume_root=root)
        )
    return descriptors


def _compile_named_volume(
    volume_id: str,
    *,
    volume_root: str | Path | None = None,
) -> dict[str, Any]:
    descriptor = _load_volume_descriptor(volume_id, volume_root=volume_root)
    volume_dir = _safe_volume_dir(volume_id, volume_root=volume_root)
    rules_path = (volume_dir / str(descriptor["corpus"]["rules_path"])).resolve()
    if rules_path.parent != volume_dir or not rules_path.is_file():
        raise ValueError(f"invalid rules_path for volume {volume_id}")
    build_config = descriptor["build"]
    with tempfile.TemporaryDirectory(prefix=f"source2agent-{volume_id}-") as work:
        build_path = Path(work) / "build"
        compile_build(
            volume_id=volume_id,
            output_dir=build_path,
            railroad_rules_path=rules_path,
            document_id=str(descriptor["document"]["document_id"]),
            seed=int(build_config["seed"]),
            group_by=str(build_config["group_by"]),
            source_revision=str(descriptor["document"]["source_revision"]),
        )
        validated = validate_build(build_path)
        inspected = inspect_build(build_path)
        return {
            "product": "Source2Agent",
            "engine": "volume2gym",
            "mode": "named-volume-compiler",
            "volume_id": volume_id,
            "validated": _jsonable(validated),
            "inspection": _jsonable(inspected),
        }


def _reference_eval_named_volume(
    volume_id: str,
    *,
    volume_root: str | Path | None = None,
) -> dict[str, Any]:
    descriptor = _load_volume_descriptor(volume_id, volume_root=volume_root)
    volume_dir = _safe_volume_dir(volume_id, volume_root=volume_root)
    rules_path = (volume_dir / str(descriptor["corpus"]["rules_path"])).resolve()
    build_config = descriptor["build"]
    with tempfile.TemporaryDirectory(prefix=f"source2agent-eval-{volume_id}-") as work:
        build_path = Path(work) / "build"
        compile_build(
            volume_id=volume_id,
            output_dir=build_path,
            railroad_rules_path=rules_path,
            document_id=str(descriptor["document"]["document_id"]),
            seed=int(build_config["seed"]),
            group_by=str(build_config["group_by"]),
            source_revision=str(descriptor["document"]["source_revision"]),
        )
        tasks = tuple(task for task in load_build(build_path).tasks if task.split is Split.TEST)
        policy = SymbolicTrainer().train(tasks)
        records = evaluate_policy(policy, tasks)
        scores = [float(record.reward_ledger.total_score or 0.0) for record in records]
        return {
            "product": "Source2Agent",
            "engine": "volume2gym",
            "mode": "symbolic-reference",
            "volume_id": volume_id,
            "model_id": policy.model_id,
            "split": "test",
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
        path = urlparse(self.path).path
        if path == "/health":
            self._send(
                200,
                {
                    "status": "healthy",
                    "product": "Source2Agent",
                    "service_mode": (
                        "custom-and-named" if _custom_inputs_enabled() else "named-volume-only"
                    ),
                    "revision": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "local"),
                    "neural_model": False,
                },
            )
            return
        if path == "/volumes":
            volumes = _available_volumes()
            self._send(
                200,
                {
                    "product": "Source2Agent",
                    "count": len(volumes),
                    "volumes": volumes,
                },
            )
            return
        if path.startswith("/volumes/") and path.count("/") == 2:
            try:
                self._send(200, _load_volume_descriptor(path.removeprefix("/volumes/")))
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                self._send(404, {"error": str(exc)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        named_match = re.fullmatch(
            r"/volumes/([a-z0-9_-]+)/(compile|reference-eval)", path
        )
        if path not in {"/compile", "/reference-eval"} and named_match is None:
            self._send(404, {"error": "not found"})
            return
        try:
            if path in {"/compile", "/reference-eval"} and not _custom_inputs_enabled():
                self._send(
                    403,
                    {
                        "error": "custom input endpoints are disabled on this deployment",
                        "named_volume_endpoints_available": True,
                    },
                )
                return
            length = int(self.headers.get("Content-Length", "0"))
            maximum = _positive_int_env(
                "SOURCE2AGENT_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES
            )
            if length < 0:
                raise ValueError("Content-Length must not be negative")
            if length > maximum:
                self._send(413, {"error": f"request body exceeds {maximum} bytes"})
                return
            payload = json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            if named_match:
                volume_id, operation = named_match.groups()
                result = _cached_named_result(
                    volume_id,
                    operation,
                    lambda: (
                        _compile_named_volume(volume_id)
                        if operation == "compile"
                        else _reference_eval_named_volume(volume_id)
                    ),
                )
            else:
                result = _run_bounded(
                    lambda: (
                        _compile(payload) if path == "/compile" else _reference_eval(payload)
                    )
                )
            self._send(200, result)
        except ServiceBusyError as exc:
            self._send(503, {"error": str(exc), "retryable": True})
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
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
