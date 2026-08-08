#!/usr/bin/env python3
"""Verify a hosted Source2Agent named-volume deployment and emit compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

VOLUME_ID = "railroad-1959-v0"


def _response_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    timeout: float = 180.0,
) -> tuple[int, dict[str, Any]]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=b"{}" if method == "POST" else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller supplies URL
        payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} did not return a JSON object")
        return response.status, payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_smoke(
    base_url: str,
    *,
    expected_revision: str | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    health_status, health = _request_json(base_url, "/health", timeout=timeout)
    descriptor_status, descriptor = _request_json(
        base_url, f"/volumes/{VOLUME_ID}", timeout=timeout
    )
    compile_status, compiled = _request_json(
        base_url,
        f"/volumes/{VOLUME_ID}/compile",
        method="POST",
        timeout=timeout,
    )
    eval_status, evaluated = _request_json(
        base_url,
        f"/volumes/{VOLUME_ID}/reference-eval",
        method="POST",
        timeout=timeout,
    )

    _require(health_status == 200, "health endpoint did not return HTTP 200")
    _require(health.get("status") == "healthy", "health status is not healthy")
    _require(health.get("product") == "Source2Agent", "unexpected health product")
    _require(health.get("neural_model") is False, "health response crossed neural boundary")
    if expected_revision is not None:
        _require(
            health.get("revision") == expected_revision,
            "deployed revision does not match --expected-revision",
        )

    _require(descriptor_status == 200, "descriptor endpoint did not return HTTP 200")
    _require(descriptor.get("volume_id") == VOLUME_ID, "unexpected descriptor volume")
    _require(
        descriptor.get("corpus", {}).get("knowledge_unit_count") == 457,
        "descriptor knowledge-unit count changed",
    )
    _require(
        descriptor.get("build", {}).get("task_count") == 2742,
        "descriptor task count changed",
    )

    validated = compiled.get("validated", {})
    _require(compile_status == 200, "compile endpoint did not return HTTP 200")
    _require(compiled.get("mode") == "named-volume-compiler", "unexpected compile mode")
    _require(validated.get("valid") is True, "hosted compile did not validate")
    _require(validated.get("knowledge_unit_count") == 457, "compiled unit count changed")
    _require(validated.get("task_count") == 2742, "compiled task count changed")
    _require(
        validated.get("split_counts") == {"train": 2190, "dev": 276, "test": 276},
        "compiled split counts changed",
    )

    _require(eval_status == 200, "reference-eval endpoint did not return HTTP 200")
    _require(evaluated.get("mode") == "symbolic-reference", "unexpected evaluation mode")
    _require(evaluated.get("split") == "test", "reference evaluation is not test-only")
    _require(evaluated.get("task_count") == 276, "reference task count changed")
    _require(evaluated.get("mean_total_score") == 1.0, "symbolic reference score changed")
    _require(evaluated.get("neural_model") is False, "evaluation crossed neural boundary")

    return {
        "schema_version": "1.0",
        "provider": "railway",
        "base_url": base_url.rstrip("/"),
        "verified_at": datetime.now(UTC).isoformat(),
        "expected_revision": expected_revision,
        "observed_revision": health.get("revision"),
        "evidence_mode": "deterministic-compiler-and-symbolic-reference",
        "neural_model": False,
        "checks": {
            "health": {
                "http_status": health_status,
                "status": health["status"],
                "service_mode": health.get("service_mode"),
                "response_sha256": _response_sha256(health),
            },
            "descriptor": {
                "http_status": descriptor_status,
                "volume_id": descriptor["volume_id"],
                "knowledge_unit_count": descriptor["corpus"]["knowledge_unit_count"],
                "task_count": descriptor["build"]["task_count"],
                "response_sha256": _response_sha256(descriptor),
            },
            "compile": {
                "http_status": compile_status,
                "valid": validated["valid"],
                "knowledge_unit_count": validated["knowledge_unit_count"],
                "task_count": validated["task_count"],
                "split_counts": validated["split_counts"],
                "response_sha256": _response_sha256(compiled),
            },
            "reference_eval": {
                "http_status": eval_status,
                "split": evaluated["split"],
                "task_count": evaluated["task_count"],
                "mean_total_score": evaluated["mean_total_score"],
                "neural_model": evaluated["neural_model"],
                "response_sha256": _response_sha256(evaluated),
            },
        },
        "limitations": [
            "This verifies a deterministic compiler and symbolic answer-key evaluator.",
            "It is not evidence of neural training, inference, learning, or generalization.",
            "The railroad OCR remains unreviewed and is not current operating instruction.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Hosted API base URL, such as https://example.up.railway.app")
    parser.add_argument("--expected-revision")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evidence = run_smoke(
        args.base_url,
        expected_revision=args.expected_revision,
        timeout=args.timeout,
    )
    rendered = json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
