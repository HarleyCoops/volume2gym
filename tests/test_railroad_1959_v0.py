import importlib.util
import sys
from pathlib import Path

import pytest

from volume2gym.profiles.railroad import RailroadProfile

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_railroad_1959_v0.py"
SPEC = importlib.util.spec_from_file_location("build_railroad_1959_v0", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def pages_fixture() -> list[str]:
    pages = [""] * 118
    pages[4] = "GENERAL RULES\nA. First gen-\neral rule.\nB. Second general rule.\n"
    pages[8] = "OPERATING RULES\n1. First numeric rule.\n4, Corrected start.\n"
    pages[12] = "ENGINE WHISTLE SIGNALS\n99.\n24\n14, A corrected rule.\n"
    pages[23] = (
        "93. First railroad variant.\n"
        "THE FOLLOWING RULE 93 APPLIES ONLY\n"
        "ON THE EXAMPLE RAILROAD.\n"
        "93. Second railroad variant.\n"
    )
    pages[114] = "992. Dispatching rule.\n1251. Legal rule.\n"
    pages[115] = "1255. Final legal rule.\nTABLE OF CONTENTS\nNoise.\n"
    return pages


def test_extracts_rules_with_corrections_exclusions_and_stable_variants():
    rules, report = MODULE.extract_rules(pages_fixture())

    assert [rule["rule_id"] for rule in rules] == [
        "A",
        "B",
        "1",
        "4",
        "14",
        "93",
        "93@p023-v2",
        "992",
        "1251",
        "1255",
    ]
    assert report["extracted_rule_count"] == 10
    assert report["compiled_task_count_expected"] == 60
    assert report["duplicate_original_rule_ids"] == {"93": 2}
    assert report["excluded_rule_like_candidates"] == [
        {"page_number": 12, "rule_id": "99", "line": "99."}
    ]
    assert rules[6]["conditions"] == [
        "THE FOLLOWING RULE 93 APPLIES ONLY ON THE EXAMPLE RAILROAD."
    ]
    assert rules[-3]["section"] == "Train Dispatchers"
    assert rules[-2]["section"] == "Legal Proceedings and Accidents"
    assert rules[-1]["section"] == "Legal Proceedings and Accidents"
    assert "Noise" not in rules[-1]["text"]


def test_profile_preserves_extraction_metadata():
    payload = {
        "rules": [
            {
                "rule_id": "4",
                "text": "A rule.",
                "metadata": {"source_sha256": "abc", "ocr_review_status": "unreviewed"},
            }
        ]
    }
    unit = RailroadProfile().import_units(payload)[0]
    assert unit.metadata["source_sha256"] == "abc"
    assert unit.metadata["ocr_review_status"] == "unreviewed"


def test_profile_rejects_non_object_metadata():
    with pytest.raises(ValueError, match="metadata for '4' must be an object"):
        RailroadProfile().import_units(
            {"rules": [{"rule_id": "4", "text": "A rule.", "metadata": "bad"}]}
        )
