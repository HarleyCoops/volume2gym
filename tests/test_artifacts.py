from __future__ import annotations

import json

import pytest

from volume2gym.artifacts import ArtifactStore, canonical_json, hash_json, sha256_file
from volume2gym.models import Citation, KnowledgeKind, KnowledgeUnit


def _unit() -> KnowledgeUnit:
    citation = Citation(document_id="manual", span_id="manual:1", quote="Stop.")
    return KnowledgeUnit(
        unit_id="manual-rule-1",
        kind=KnowledgeKind.RULE,
        title="Stop rule",
        text="Stop.",
        family="movement",
        citations=(citation,),
        required_actions=("Stop.",),
    )


def test_canonical_json_and_hash_ignore_mapping_insertion_order() -> None:
    left = {"z": 1, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "z": 1}

    assert canonical_json(left) == canonical_json(right)
    assert hash_json(left) == hash_json(right)
    assert canonical_json(left) == '{"nested":{"a":1,"b":2},"z":1}'


def test_jsonl_round_trip_hash_and_record_count(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    reference = store.write_jsonl("knowledge/units.jsonl", [_unit(), _unit()])

    assert reference.record_count == 2
    assert reference.sha256 == sha256_file(tmp_path / reference.path)
    assert list(store.read_jsonl(reference.path, KnowledgeUnit)) == [_unit(), _unit()]
    store.validate(reference)


def test_atomic_write_does_not_replace_valid_artifact_on_serialization_error(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.write_json("manifest.json", {"valid": True})

    with pytest.raises(TypeError, match="not JSON serializable"):
        store.write_json("manifest.json", {"bad": object()})

    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {
        "valid": True
    }
    assert not list(tmp_path.glob(".manifest.json.*"))


def test_store_rejects_path_escape_and_detects_tampering(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        store.write_json("../outside.json", {})

    reference = store.write_json("value.json", {"value": 1})
    (tmp_path / "value.json").write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.validate(reference)
