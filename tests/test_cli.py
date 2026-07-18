from __future__ import annotations

import json

from volume2gym.cli import main
from volume2gym.models import Citation, KnowledgeKind, KnowledgeUnit


def _write_units(path) -> None:
    citation = Citation(document_id="manual", span_id="manual:1", quote="Stop.")
    unit = KnowledgeUnit(
        unit_id="rule-1",
        kind=KnowledgeKind.RULE,
        title="Stop",
        text="Stop.",
        family="movement",
        citations=(citation,),
        required_actions=("Stop.",),
    )
    path.write_text(json.dumps([unit.model_dump(mode="json")]), encoding="utf-8")


def test_compile_validate_and_inspect_commands(tmp_path, capsys) -> None:
    units = tmp_path / "units.json"
    build = tmp_path / "build"
    _write_units(units)

    assert (
        main(
            [
                "compile",
                "--volume-id",
                "cli-volume",
                "--output",
                str(build),
                "--units",
                str(units),
                "--seed",
                "7",
            ]
        )
        == 0
    )
    compiled = json.loads(capsys.readouterr().out)
    assert compiled["valid"] is True
    assert compiled["task_count"] == 6

    assert main(["validate", str(build)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["build_id"] == compiled["build_id"]

    assert main(["inspect-artifacts", str(build)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["task_family_counts"]["standard_operation"] == 1
    assert len(inspected["artifacts"]) == 5


def test_cli_returns_nonzero_and_clear_error_for_missing_build(tmp_path, capsys) -> None:
    assert main(["validate", str(tmp_path / "missing")]) == 2
    captured = capsys.readouterr()
    assert "v2g: error:" in captured.err
    assert "manifest.json" in captured.err


def test_export_and_symbolic_reference_eval_close_the_local_loop(tmp_path, capsys) -> None:
    units = tmp_path / "units.json"
    build = tmp_path / "build"
    _write_units(units)
    assert (
        main(
            [
                "compile",
                "--volume-id",
                "loop-volume",
                "--output",
                str(build),
                "--units",
                str(units),
            ]
        )
        == 0
    )
    capsys.readouterr()

    for export_format in ("sft", "grpo"):
        output = tmp_path / f"{export_format}.jsonl"
        assert (
            main(
                [
                    "export",
                    str(build),
                    "--format",
                    export_format,
                    "--output",
                    str(output),
                    "--split",
                    "all",
                ]
            )
            == 0
        )
        exported = json.loads(capsys.readouterr().out)
        assert exported["format"] == export_format
        assert exported["record_count"] == 6
        assert len(exported["sha256"]) == 64
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 6
        assert ("messages" in rows[0]) is (export_format == "sft")
        assert ("answer_key" in rows[0]) is (export_format == "grpo")

    evaluation_dir = tmp_path / "reference-eval"
    assert (
        main(
            [
                "reference-eval",
                str(build),
                "--output",
                str(evaluation_dir),
                "--split",
                "all",
            ]
        )
        == 0
    )
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["reference_type"] == "symbolic-answer-key"
    assert evaluation["neural_model"] is False
    assert evaluation["count"] == 6
    assert len(evaluation["artifacts"]) == 3
    assert (evaluation_dir / "responses.jsonl").is_file()
    assert (evaluation_dir / "reward_ledgers.jsonl").is_file()
    assert (evaluation_dir / "evaluation.json").is_file()
