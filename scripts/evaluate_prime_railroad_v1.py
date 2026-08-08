#!/usr/bin/env python3
"""Create deterministic model-free verifiers.v1 reference evaluation artifacts."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from railroad_1959_v1 import Railroad1959Config, Railroad1959Taskset, reference_trace

REPO = Path(__file__).parents[1]
OUTPUT_DIR = REPO / "evidence" / "prime" / "railroad-1959-v1"
TASKSET_DIR = (
    REPO
    / "environments"
    / "railroad_1959_v1"
    / "railroad_1959_v1"
    / "data"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def deterministic_gzip(value: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=buffer,
        mtime=0,
    ) as archive:
        archive.write(value)
    return buffer.getvalue()


async def build_payloads() -> dict[str, bytes]:
    manifest_path = TASKSET_DIR / "taskset-manifest.json"
    taskset_manifest = json.loads(manifest_path.read_text())
    tasks = list(Railroad1959Taskset(Railroad1959Config(split="test")))
    traces: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    rewards: list[float] = []
    families: Counter[str] = Counter()
    fidelity: list[float] = []
    for task in tasks:
        trace = reference_trace(task)
        await task.score(trace)
        ledger = trace.info["source2agent_reward_ledger"]
        traces.append(trace.to_record())
        ledgers.append(ledger)
        rewards.append(trace.reward)
        families[task.data.task_family] += 1
        fidelity.append(trace.metrics["reward.reference_answer_fidelity_unscored"])
    trace_raw = ("\n".join(canonical_json(row) for row in traces) + "\n").encode()
    ledger_raw = ("\n".join(canonical_json(row) for row in ledgers) + "\n").encode()
    trace_gzip = deterministic_gzip(trace_raw)
    ledger_gzip = deterministic_gzip(ledger_raw)
    summary = {
        "schema_version": "0.1",
        "environment_id": "railroad-1959-v1",
        "environment_version": "0.1.0",
        "evaluation_id": "railroad-1959-v1-symbolic-reference",
        "mode": "symbolic-answer-key",
        "split": "test",
        "task_count": len(tasks),
        "task_family_counts": dict(sorted(families.items())),
        "mean_total_reward": sum(rewards) / len(rewards),
        "min_total_reward": min(rewards),
        "max_total_reward": max(rewards),
        "mean_reference_answer_fidelity_unscored": sum(fidelity) / len(fidelity),
        "verifiers_version": "0.3.0",
        "verifiers_release_commit": "0a4d872f021022310a08ec213a25f4efb4a0244a",
        "harness": "null",
        "runtime": "subprocess-contract",
        "model_id": "symbolic-answer-key",
        "neural_model": False,
        "neural_learning_claimed": False,
        "hosted": False,
        "hosted_credits_spent": False,
        "checkpoint_published": False,
        "taskset_manifest_sha256": sha256(manifest_path.read_bytes()),
        "taskset_jsonl_sha256": taskset_manifest["taskset_jsonl_sha256"],
        "taskset_gzip_sha256": taskset_manifest["taskset_gzip_sha256"],
        "reward_verifier_id": "volume2gym.deterministic-composite",
        "reward_verifier_version": "1",
        "traces_jsonl_sha256": sha256(trace_raw),
        "traces_gzip_sha256": sha256(trace_gzip),
        "reward_ledgers_jsonl_sha256": sha256(ledger_raw),
        "reward_ledgers_gzip_sha256": sha256(ledger_gzip),
        "limitations": [
            "The policy copies the answer key; no model inference or learning occurred.",
            (
                "Held-out prompts include their source excerpt, so the split tests "
                "in-context source following rather than memorized or cross-volume "
                "knowledge generalization."
            ),
            "The OCR corpus is unreviewed and is not current railroad instruction.",
            (
                "All extracted action, prohibition, procedure, and terminology keys "
                "are empty; only applicable-rule reward varies."
            ),
            "Reference-answer fidelity is recorded as an unscored metric.",
            "The local subprocess runtime does not enforce network isolation.",
            (
                "The symbolic reference traces were constructed and scored locally; "
                "they did not execute a model-facing harness or rollout runtime."
            ),
        ],
        "entire": {
            "agent_integration": "pi",
            "session_recorded": False,
            "checkpoint_published": False,
            "reason": (
                "Entire CLI was unavailable and public checkpoint publication lacks "
                "an explicit privacy approval."
            ),
        },
    }
    return {
        "traces.jsonl.gz": trace_gzip,
        "reward-ledgers.jsonl.gz": ledger_gzip,
        "summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    }


async def run(check: bool) -> int:
    payloads = await build_payloads()
    if check:
        for filename, expected in payloads.items():
            path = OUTPUT_DIR / filename
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"Prime reference artifact is missing or stale: {path}")
        print("railroad-1959-v1 symbolic reference artifacts: reproducible")
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        (OUTPUT_DIR / filename).write_bytes(payload)
        print(OUTPUT_DIR / filename)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
