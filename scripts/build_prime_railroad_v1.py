#!/usr/bin/env python3
"""Build the exact packaged taskset for the railroad verifiers.v1 environment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from volume2gym.pipeline import compile_build

REPO = Path(__file__).parents[1]
RULES = REPO / "volumes" / "railroad-1959-v0" / "rules.json"
OUTPUT_DIR = (
    REPO
    / "environments"
    / "railroad_1959_v1"
    / "railroad_1959_v1"
    / "data"
)
SOURCE_REVISION = "cd7cfd8bab3d9d9c33446c971f5df8276e5a29f4"
SOURCE_SHA256 = "c96a60c2b20e7b34d9bd689d57b2ec5b8c71362545b889c479bdf04fe6444350"
VERIFIERS_VERSION = "0.3.0"
VERIFIERS_COMMIT = "0a4d872f021022310a08ec213a25f4efb4a0244a"
RESPONSE_CONTRACT = """

Response contract:
Return only one JSON object with exactly these fields:
{"applicable_rules":["source unit id"],"situation_type":"task family",\
"required_actions":[],"forbidden_actions":[],"procedure_order":[],\
"final_answer":"source-grounded answer"}
Do not add Markdown fences or text outside the JSON object.
""".strip()


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


def build_payloads() -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory(prefix="source2agent-prime-taskset-") as work:
        build = compile_build(
            volume_id="railroad-1959-v0",
            output_dir=Path(work) / "build",
            railroad_rules_path=RULES,
            document_id="railroad-1959",
            seed=1959,
            group_by="knowledge_unit",
            source_revision=SOURCE_REVISION,
        )
        records: list[dict[str, Any]] = []
        split_groups: dict[str, set[str]] = defaultdict(set)
        family_counts: Counter[str] = Counter()
        nonempty_keys: Counter[str] = Counter()
        for idx, task in enumerate(build.tasks):
            if task.split is None:
                raise ValueError(f"task {task.task_id} has no split")
            if len(task.knowledge_unit_ids) != 1:
                raise ValueError(f"task {task.task_id} is not a single-unit structural group")
            group_id = task.knowledge_unit_ids[0]
            split = task.split.value
            split_groups[split].add(group_id)
            family_counts[task.task_family.value] += 1
            answer_key = task.answer_key.model_dump(mode="json")
            for field in (
                "required_actions",
                "forbidden_actions",
                "procedure_order",
                "terms",
            ):
                nonempty_keys[field] += int(bool(answer_key[field]))
            records.append(
                {
                    "idx": idx,
                    "task_id": task.task_id,
                    "prompt": f"{task.prompt}\n\n{RESPONSE_CONTRACT}",
                    "split": split,
                    "task_family": task.task_family.value,
                    "knowledge_unit_ids": list(task.knowledge_unit_ids),
                    "group_id": group_id,
                    "answer_contract": task.answer_contract.model_dump(mode="json"),
                    "answer_key": answer_key,
                    "citations": [
                        citation.model_dump(mode="json") for citation in task.citations
                    ],
                }
            )

        intersections = {
            "train_dev": sorted(split_groups["train"] & split_groups["dev"]),
            "train_test": sorted(split_groups["train"] & split_groups["test"]),
            "dev_test": sorted(split_groups["dev"] & split_groups["test"]),
        }
        if any(intersections.values()):
            raise ValueError("structural holdout leakage detected")

        raw = ("\n".join(canonical_json(record) for record in records) + "\n").encode()
        compressed = deterministic_gzip(raw)
        source_task_ref = next(
            ref for ref in build.manifest.outputs if ref.path == "tasks/all.jsonl"
        )
        split_counts = Counter(record["split"] for record in records)
        manifest = {
            "schema_version": "0.1",
            "environment_id": "railroad-1959-v1",
            "environment_hub_id": None,
            "environment_version": "0.1.0",
            "verifiers_version": VERIFIERS_VERSION,
            "verifiers_release_commit": VERIFIERS_COMMIT,
            "source2agent_build_id": build.manifest.build_id,
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "source_rules_sha256": sha256(RULES.read_bytes()),
            "source_task_artifact_sha256": source_task_ref.sha256,
            "taskset_jsonl_sha256": sha256(raw),
            "taskset_gzip_sha256": sha256(compressed),
            "task_count": len(records),
            "split_counts": dict(sorted(split_counts.items())),
            "task_family_counts": dict(sorted(family_counts.items())),
            "holdout_contract": {
                "group_by": "knowledge_unit",
                "group_counts": {
                    split: len(groups) for split, groups in sorted(split_groups.items())
                },
                "cross_split_intersections": intersections,
                "leakage_valid": True,
            },
            "reward_contract": {
                "verifier_id": "volume2gym.deterministic-composite",
                "verifier_version": "1",
                "safety_hard_gate": True,
                "safety_threshold": 1.0,
                "weights": {
                    "safety_critical_required_actions": 0.40,
                    "forbidden_unsafe_actions_absent": 0.20,
                    "correct_applicable_rule_cited": 0.15,
                    "procedure_order_correct": 0.15,
                    "domain_terminology_correct": 0.10,
                },
                "reference_answer_fidelity": "unscored_metric",
                "situation_type": "schema_only_unscored",
            },
            "answer_key_nonempty_counts": dict(sorted(nonempty_keys.items())),
            "runtime_contract": {
                "harness": "null",
                "runtime": "subprocess",
                "max_turns": 1,
                "network_policy": "unenforced-by-subprocess-runtime",
                "hosted": False,
            },
            "neural_model": False,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        return compressed, manifest_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    compressed, manifest = build_payloads()
    taskset_path = OUTPUT_DIR / "taskset.jsonl.gz"
    manifest_path = OUTPUT_DIR / "taskset-manifest.json"
    if args.check:
        if not taskset_path.is_file() or taskset_path.read_bytes() != compressed:
            raise SystemExit("packaged railroad taskset is missing or stale")
        if not manifest_path.is_file() or manifest_path.read_bytes() != manifest:
            raise SystemExit("packaged railroad taskset manifest is missing or stale")
        print("railroad-1959-v1 packaged taskset: reproducible")
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    taskset_path.write_bytes(compressed)
    manifest_path.write_bytes(manifest)
    print(taskset_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
