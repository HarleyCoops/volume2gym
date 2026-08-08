import asyncio
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from railroad_1959_v1 import (
    Railroad1959Config,
    Railroad1959Taskset,
    reference_response,
    reference_trace,
    score_response,
)

from volume2gym.models import Task as CoreTask
from volume2gym.verifier import DeterministicVerifier

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "evidence" / "prime" / "railroad-1959-v1"
DATA = (
    ROOT
    / "environments"
    / "railroad_1959_v1"
    / "railroad_1959_v1"
    / "data"
)


def test_exact_taskset_hashes_counts_and_structural_holdouts():
    manifest = json.loads((DATA / "taskset-manifest.json").read_text())
    compressed = (DATA / "taskset.jsonl.gz").read_bytes()
    raw = gzip.decompress(compressed)
    assert compressed[9] == 255
    assert hashlib.sha256(compressed).hexdigest() == manifest["taskset_gzip_sha256"]
    assert hashlib.sha256(raw).hexdigest() == manifest["taskset_jsonl_sha256"]

    taskset = list(Railroad1959Taskset(Railroad1959Config(split="all")))
    assert len(taskset) == 2742
    counts = defaultdict(int)
    groups = defaultdict(set)
    for task in taskset:
        counts[task.data.split] += 1
        groups[task.data.group_id].add(task.data.split)
    assert dict(counts) == {"train": 2190, "dev": 276, "test": 276}
    assert all(len(splits) == 1 for splits in groups.values())
    assert len(groups) == 457


def test_evidence_binds_taskset_reward_and_gold_validation():
    manifest_path = DATA / "taskset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary_path = EVIDENCE / "summary.json"
    summary = json.loads(summary_path.read_text())
    gold_path = EVIDENCE / "gold-validation-summary.json"
    gold = json.loads(gold_path.read_text())
    volume = json.loads((ROOT / "volumes" / "railroad-1959-v0" / "volume.json").read_text())
    prime = volume["prime_lab"]

    assert summary["taskset_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert summary["taskset_jsonl_sha256"] == manifest["taskset_jsonl_sha256"]
    assert summary["taskset_gzip_sha256"] == manifest["taskset_gzip_sha256"]
    assert summary["reward_verifier_id"] == prime["reward_verifier_id"]
    assert summary["reward_verifier_version"] == prime["reward_verifier_version"]
    assert prime["reference_eval_summary_sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    for field, filename in (
        ("reference_eval_traces_gzip_sha256", "traces.jsonl.gz"),
        ("reference_eval_ledgers_gzip_sha256", "reward-ledgers.jsonl.gz"),
    ):
        payload = (EVIDENCE / filename).read_bytes()
        assert payload[9] == 255
        assert prime[field] == hashlib.sha256(payload).hexdigest()
    assert prime["gold_validation_summary_sha256"] == hashlib.sha256(
        gold_path.read_bytes()
    ).hexdigest()
    assert gold["outcomes"] == {
        "error": 0,
        "invalid": 0,
        "missing": 0,
        "timeout": 0,
        "valid": 2742,
    }


def test_v1_reward_matches_core_verifier_for_every_symbolic_reference():
    taskset = list(Railroad1959Taskset(Railroad1959Config(split="all")))
    core_tasks = {
        task.task_id: task
        for task in _load_core_tasks_from_packaged_artifact()
    }
    verifier = DeterministicVerifier()
    for task in taskset:
        response = reference_response(task)
        response_id = f"{task.data.task_id}.parity"
        actual = score_response(task.data, response, response_id=response_id)
        expected = verifier.verify(
            core_tasks[task.data.task_id], response, response_id=response_id
        ).model_dump(mode="json")
        assert actual == expected


def test_reference_trace_is_real_v1_trace_and_explicitly_non_neural():
    task = next(iter(Railroad1959Taskset(Railroad1959Config(split="test"))))
    trace = reference_trace(task)
    asyncio.run(task.score(trace))
    assert trace.reward == 1.0
    assert trace.info["neural_model"] is False
    assert trace.info["hosted_credits_spent"] is False
    assert trace.nodes[-1].sampled is False
    assert trace.num_turns == 0
    assert trace.info["symbolic_response"] == reference_response(task)
    assert trace.agent.config.harness.id == "null"
    assert trace.agent.config.runtime.type == "subprocess"
    assert trace.metrics["reward.reference_answer_fidelity_unscored"] == 1.0
    assert trace.to_record()["version"] == 1


def test_invalid_response_scores_zero_and_reward_limit_is_visible():
    task = next(iter(Railroad1959Taskset(Railroad1959Config(split="test"))))
    invalid = score_response(task.data, "not json", response_id="invalid")
    assert invalid["total_score"] == 0.0
    assert invalid["gate_multiplier"] == 0.0

    # Evidence limit: this v0 extraction has no action/procedure/term keys, so a
    # schema-valid response citing only the applicable rule still earns 1.0.
    rule_only = {
        "applicable_rules": list(task.data.answer_key["applicable_unit_ids"]),
        "situation_type": None,
        "required_actions": [],
        "forbidden_actions": [],
        "procedure_order": [],
        "final_answer": "",
    }
    ledger = score_response(task.data, rule_only, response_id="rule-only")
    assert ledger["total_score"] == 1.0


def test_v1_reward_matches_core_verifier_on_populated_contract_paths():
    base = next(iter(Railroad1959Taskset(Railroad1959Config(split="test"))))
    answer_key = {
        "applicable_unit_ids": ["rule-1", "rule-2"],
        "required_actions": ["stop", "inspect"],
        "forbidden_actions": ["proceed", "bypass"],
        "procedure_order": ["stop", "secure", "inspect"],
        "terms": ["flag protection", "main track"],
        "reference_answer": "Use flag protection on the main track.",
        "citations": [],
    }
    data = base.data.model_copy(
        update={
            "task_id": "synthetic-populated-contract",
            "name": "synthetic-populated-contract",
            "knowledge_unit_ids": ("rule-1", "rule-2"),
            "answer_key": answer_key,
        }
    )
    core = CoreTask.model_validate(
        {
            "task_id": data.task_id,
            "prompt": data.prompt,
            "task_family": data.task_family,
            "knowledge_unit_ids": data.knowledge_unit_ids,
            "answer_contract": data.answer_contract,
            "answer_key": answer_key,
            "citations": [],
            "split": data.split,
        }
    )
    responses = [
        {
            "applicable_rules": ["rule-1", "rule-2"],
            "situation_type": "intentionally unscored",
            "required_actions": ["stop", "inspect"],
            "forbidden_actions": ["proceed", "bypass"],
            "procedure_order": ["stop", "secure", "inspect"],
            "final_answer": "Use flag protection on the main track.",
        },
        {
            "applicable_rules": ["rule-1"],
            "required_actions": ["stop"],
            "forbidden_actions": ["proceed"],
            "procedure_order": ["inspect", "secure", "stop"],
            "final_answer": "Use flag protection.",
        },
        {
            "applicable_rules": ["rule-1", "rule-2"],
            "required_actions": ["stop", "inspect", "proceed"],
            "forbidden_actions": ["proceed", "bypass"],
            "procedure_order": ["stop", "secure", "inspect"],
            "final_answer": "Use flag protection on the main track.",
        },
        "not json",
    ]
    verifier = DeterministicVerifier()
    for index, response in enumerate(responses):
        response_id = f"synthetic-{index}"
        assert score_response(data, response, response_id=response_id) == (
            verifier.verify(core, response, response_id=response_id).model_dump(
                mode="json"
            )
        )


def _load_core_tasks_from_packaged_artifact() -> list[CoreTask]:
    rows = [
        json.loads(line)
        for line in gzip.decompress((DATA / "taskset.jsonl.gz").read_bytes()).splitlines()
        if line
    ]
    return [
        CoreTask.model_validate(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "task_family": row["task_family"],
                "knowledge_unit_ids": row["knowledge_unit_ids"],
                "answer_contract": row["answer_contract"],
                "answer_key": row["answer_key"],
                "citations": row["citations"],
                "split": row["split"],
            }
        )
        for row in rows
    ]
