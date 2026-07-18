"""Dependency-free command line interface for compiling and auditing builds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, sha256_file
from .exporters import write_grpo_jsonl, write_sft_jsonl
from .models import Split, Task
from .pipeline import compile_build, inspect_build, load_build, validate_build
from .trainers import SymbolicTrainer, evaluate_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2g",
        description="Compile structured volumes into source-grounded learning environments.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    compile_parser = subcommands.add_parser(
        "compile",
        help="compile canonical units or legacy railroad JSON into a build",
    )
    compile_parser.add_argument("--volume-id", required=True)
    compile_parser.add_argument("--output", required=True, type=Path)
    source = compile_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--units",
        "--canonical-units",
        dest="canonical_units_path",
        type=Path,
        help="canonical KnowledgeUnit JSON or JSONL",
    )
    source.add_argument(
        "--railroad-rules",
        dest="railroad_rules_path",
        type=Path,
        help="legacy RailroadRule JSON or JSONL",
    )
    compile_parser.add_argument("--railroad-tasks", type=Path)
    compile_parser.add_argument("--document-id")
    compile_parser.add_argument("--seed", type=int, default=0)
    compile_parser.add_argument(
        "--group-by",
        choices=("rule_family", "knowledge_unit"),
        default="rule_family",
    )
    compile_parser.add_argument("--source-revision")

    validate_parser = subcommands.add_parser(
        "validate",
        help="validate schemas, hashes, split membership, and source-unit references",
    )
    validate_parser.add_argument("build_dir", type=Path)

    inspect_parser = subcommands.add_parser(
        "inspect-artifacts",
        aliases=["inspect"],
        help="validate a build and print its artifact/task summary",
    )
    inspect_parser.add_argument("build_dir", type=Path)

    export_parser = subcommands.add_parser(
        "export",
        help="export a compiled task split as trainer-ready SFT or GRPO JSONL",
    )
    export_parser.add_argument("build_dir", type=Path)
    export_parser.add_argument("--format", choices=("sft", "grpo"), required=True)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        default="train",
    )

    reference_parser = subcommands.add_parser(
        "reference-eval",
        help="run the transparent symbolic answer-key reference and persist its ledgers",
    )
    reference_parser.add_argument("build_dir", type=Path)
    reference_parser.add_argument("--output", required=True, type=Path)
    reference_parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "compile":
            compile_build(
                volume_id=arguments.volume_id,
                output_dir=arguments.output,
                canonical_units_path=arguments.canonical_units_path,
                railroad_rules_path=arguments.railroad_rules_path,
                railroad_tasks_path=arguments.railroad_tasks,
                document_id=arguments.document_id,
                seed=arguments.seed,
                group_by=arguments.group_by,
                source_revision=arguments.source_revision,
            )
            result = validate_build(arguments.output)
        elif arguments.command == "validate":
            result = validate_build(arguments.build_dir)
        elif arguments.command in {"inspect-artifacts", "inspect"}:
            result = inspect_build(arguments.build_dir)
        elif arguments.command == "export":
            tasks = _selected_tasks(arguments.build_dir, arguments.split)
            writer = write_sft_jsonl if arguments.format == "sft" else write_grpo_jsonl
            output = writer(tasks, arguments.output).resolve()
            result = {
                "format": arguments.format,
                "path": str(output),
                "record_count": len(tasks),
                "sha256": sha256_file(output),
                "split": arguments.split,
            }
        elif arguments.command == "reference-eval":
            tasks = _selected_tasks(arguments.build_dir, arguments.split)
            policy = SymbolicTrainer().train(tasks)
            records = evaluate_policy(policy, tasks)
            store = ArtifactStore(arguments.output)
            response_ref = store.write_jsonl(
                "responses.jsonl", (record.response for record in records)
            )
            ledger_ref = store.write_jsonl(
                "reward_ledgers.jsonl", (record.reward_ledger for record in records)
            )
            scores = [float(record.reward_ledger.total_score or 0.0) for record in records]
            summary = {
                "reference_type": "symbolic-answer-key",
                "neural_model": False,
                "model_id": policy.model_id,
                "split": arguments.split,
                "count": len(records),
                "mean_total_score": sum(scores) / len(scores),
                "component_means": _component_means(records),
            }
            summary_ref = store.write_json("evaluation.json", summary)
            result = {
                **summary,
                "output_dir": str(store.root),
                "artifacts": [
                    response_ref.model_dump(mode="json"),
                    ledger_ref.model_dump(mode="json"),
                    summary_ref.model_dump(mode="json"),
                ],
            }
        else:  # pragma: no cover - argparse protects the dispatch table
            parser.error(f"unknown command: {arguments.command}")
            return 2
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"v2g: error: {exc}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _selected_tasks(build_dir: Path, split_name: str) -> tuple[Task, ...]:
    tasks = load_build(build_dir).tasks
    if split_name == "all":
        selected = tasks
    else:
        split = Split(split_name)
        selected = tuple(task for task in tasks if task.split is split)
    if not selected:
        raise ValueError(f"build contains no tasks in split {split_name!r}")
    return selected


def _component_means(records: Sequence[Any]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for component in record.reward_ledger.components:
            values[component.component_id].append(component.score)
    return {
        component_id: sum(scores) / len(scores)
        for component_id, scores in sorted(values.items())
    }


__all__ = ["build_parser", "main"]
