"""Deterministic orchestration from normalized volume records to a compiled gym build."""

from __future__ import annotations

import json
import mimetypes
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import ArtifactStore, hash_json, sha256_file
from .compiler import TEMPLATE_VERSION, TemplateCompiler
from .models import ArtifactRef, BuildManifest, KnowledgeUnit, Split, Task
from .profiles import RailroadProfile
from .splitter import GroupedSplitter

GroupMode = Literal["rule_family", "knowledge_unit"]
REQUIRED_OUTPUTS = (
    "knowledge/units.jsonl",
    "tasks/all.jsonl",
    "splits/train.jsonl",
    "splits/dev.jsonl",
    "splits/test.jsonl",
)


@dataclass(frozen=True, slots=True)
class BuildResult:
    root: Path
    manifest: BuildManifest
    manifest_ref: ArtifactRef
    knowledge_units: tuple[KnowledgeUnit, ...]
    tasks: tuple[Task, ...]


def compile_build(
    *,
    volume_id: str,
    output_dir: str | Path,
    units: Iterable[KnowledgeUnit] | None = None,
    canonical_units_path: str | Path | None = None,
    railroad_rules_path: str | Path | None = None,
    railroad_tasks_path: str | Path | None = None,
    document_id: str | None = None,
    seed: int = 0,
    group_by: GroupMode = "rule_family",
    source_revision: str | None = None,
) -> BuildResult:
    """Compile canonical units or legacy railroad JSON into a complete build.

    Exactly one of ``units``, ``canonical_units_path``, and
    ``railroad_rules_path`` must be supplied. Legacy tasks are imported in
    addition to the six deterministic compiler tasks per source unit.
    """

    volume_id = volume_id.strip()
    if not volume_id:
        raise ValueError("volume_id must not be empty")
    selected = sum(
        item is not None for item in (units, canonical_units_path, railroad_rules_path)
    )
    if selected != 1:
        raise ValueError(
            "provide exactly one of units, canonical_units_path, or railroad_rules_path"
        )
    if railroad_tasks_path is not None and railroad_rules_path is None:
        raise ValueError("railroad_tasks_path requires railroad_rules_path")
    if group_by not in ("rule_family", "knowledge_unit"):
        raise ValueError("group_by must be 'rule_family' or 'knowledge_unit'")

    normalized_units: list[KnowledgeUnit]
    imported_tasks: list[Task] = []
    input_refs: list[ArtifactRef] = []

    if units is not None:
        normalized_units = [
            item if isinstance(item, KnowledgeUnit) else KnowledgeUnit.model_validate(item)
            for item in units
        ]
    elif canonical_units_path is not None:
        path = Path(canonical_units_path)
        normalized_units = load_canonical_units(path)
        input_refs.append(_input_reference(path))
    else:
        rules_path = Path(railroad_rules_path or "")
        rules_payload = _load_records(rules_path, wrapper="rules")
        input_refs.append(_input_reference(rules_path))
        profile = RailroadProfile()
        normalized_units = profile.import_units(
            rules_payload,
            document_id=document_id or volume_id,
        )
        if railroad_tasks_path is not None:
            tasks_path = Path(railroad_tasks_path)
            tasks_payload = _load_records(tasks_path, wrapper="tasks")
            input_refs.append(_input_reference(tasks_path))
            imported_tasks = profile.import_tasks(
                tasks_payload,
                units=normalized_units,
                document_id=document_id or volume_id,
            )

    normalized_units = sorted(normalized_units, key=lambda unit: unit.unit_id)
    _require_unique((unit.unit_id for unit in normalized_units), "knowledge unit")
    if not normalized_units:
        raise ValueError("a build requires at least one knowledge unit")

    generated_tasks = TemplateCompiler().compile(normalized_units)
    all_tasks = sorted((*generated_tasks, *imported_tasks), key=lambda task: task.task_id)
    _require_unique((task.task_id for task in all_tasks), "task")
    known_units = {unit.unit_id for unit in normalized_units}
    for task in all_tasks:
        missing = set(task.knowledge_unit_ids) - known_units
        if missing:
            raise ValueError(
                f"task {task.task_id!r} references unknown units: {', '.join(sorted(missing))}"
            )

    splitter = GroupedSplitter(seed=seed, group_by=group_by)
    assigned_tasks = sorted(splitter.split(all_tasks), key=lambda task: task.task_id)
    partitions: dict[Split, list[Task]] = {split: [] for split in Split}
    for task in assigned_tasks:
        if task.split is None:  # pragma: no cover - protected by GroupedSplitter
            raise RuntimeError(f"task {task.task_id!r} has no split")
        partitions[task.split].append(task)

    store = ArtifactStore(output_dir)
    output_refs = [store.write_jsonl("knowledge/units.jsonl", normalized_units)]
    output_refs.append(store.write_jsonl("tasks/all.jsonl", assigned_tasks))
    for split in Split:
        output_refs.append(
            store.write_jsonl(f"splits/{split.value}.jsonl", partitions[split])
        )

    fingerprint = {
        "volume_id": volume_id,
        "compiler_version": TEMPLATE_VERSION,
        "seed": seed,
        "group_by": group_by,
        "source_revision": source_revision,
        "inputs": [reference.sha256 for reference in input_refs],
        "outputs": [reference.sha256 for reference in output_refs],
    }
    build_id = f"{volume_id}-{hash_json(fingerprint)[:16]}"
    manifest = BuildManifest(
        build_id=build_id,
        volume_id=volume_id,
        compiler_version=TEMPLATE_VERSION,
        seed=seed,
        inputs=tuple(input_refs),
        outputs=tuple(output_refs),
        source_revision=source_revision,
        metadata={
            "group_by": group_by,
            "knowledge_unit_count": len(normalized_units),
            "task_count": len(assigned_tasks),
            "generated_task_count": len(generated_tasks),
            "imported_task_count": len(imported_tasks),
        },
    )
    manifest_ref = store.write_json("manifest.json", manifest)
    return BuildResult(
        root=store.root,
        manifest=manifest,
        manifest_ref=manifest_ref,
        knowledge_units=tuple(normalized_units),
        tasks=tuple(assigned_tasks),
    )


def load_canonical_units(path: str | Path) -> list[KnowledgeUnit]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        values = _read_jsonl(source)
    else:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("knowledge_units", payload.get("units"))
        if not isinstance(payload, list):
            raise ValueError("canonical unit JSON must be a list or contain 'knowledge_units'")
        values = payload
    return [KnowledgeUnit.model_validate(value) for value in values]


def load_build(build_dir: str | Path) -> BuildResult:
    store = ArtifactStore(build_dir)
    manifest = store.read_json("manifest.json", BuildManifest)
    units = tuple(store.read_jsonl("knowledge/units.jsonl", KnowledgeUnit))
    tasks = tuple(store.read_jsonl("tasks/all.jsonl", Task))
    manifest_path = store.resolve("manifest.json")
    manifest_ref = ArtifactRef(
        path="manifest.json",
        sha256=sha256_file(manifest_path),
        media_type="application/json",
        record_count=1,
    )
    return BuildResult(store.root, manifest, manifest_ref, units, tasks)


def validate_build(build_dir: str | Path) -> dict[str, Any]:
    """Validate hashes, schemas, split contents, references, and leakage."""

    store = ArtifactStore(build_dir)
    manifest = store.read_json("manifest.json", BuildManifest)
    references = {reference.path: reference for reference in manifest.outputs}
    missing_outputs = sorted(set(REQUIRED_OUTPUTS) - set(references))
    if missing_outputs:
        raise ValueError("manifest is missing outputs: " + ", ".join(missing_outputs))
    # Verify bytes before parsing them. A modified artifact should always be
    # reported as a provenance failure, even when the modification also makes
    # its schema invalid.
    for reference in manifest.outputs:
        store.validate(reference)
    result = load_build(store.root)

    unit_ids = [unit.unit_id for unit in result.knowledge_units]
    _require_unique(unit_ids, "knowledge unit")
    known_units = set(unit_ids)
    _require_unique((task.task_id for task in result.tasks), "task")

    split_records: dict[Split, tuple[Task, ...]] = {}
    for split in Split:
        records = tuple(store.read_jsonl(f"splits/{split.value}.jsonl", Task))
        if any(task.split is not split for task in records):
            raise ValueError(f"splits/{split.value}.jsonl contains an incorrectly assigned task")
        split_records[split] = records

    combined_ids = [task.task_id for split in Split for task in split_records[split]]
    if Counter(combined_ids) != Counter(task.task_id for task in result.tasks):
        raise ValueError("split files do not contain exactly the tasks in tasks/all.jsonl")

    for task in result.tasks:
        if task.split is None:
            raise ValueError(f"task {task.task_id!r} has no split")
        missing = set(task.knowledge_unit_ids) - known_units
        if missing:
            raise ValueError(
                f"task {task.task_id!r} references unknown units: {', '.join(sorted(missing))}"
            )

    _validate_group_leakage(result.tasks, str(result.manifest.metadata.get("group_by", "")))
    counts = {split.value: len(split_records[split]) for split in Split}
    return {
        "valid": True,
        "build_id": result.manifest.build_id,
        "volume_id": result.manifest.volume_id,
        "knowledge_unit_count": len(result.knowledge_units),
        "task_count": len(result.tasks),
        "split_counts": counts,
        "artifact_count": len(result.manifest.outputs),
    }


def inspect_build(build_dir: str | Path) -> dict[str, Any]:
    validation = validate_build(build_dir)
    result = load_build(build_dir)
    store = ArtifactStore(result.root)
    family_counts = Counter(task.task_family.value for task in result.tasks)
    validation.update(
        {
            "compiler_version": result.manifest.compiler_version,
            "seed": result.manifest.seed,
            "task_family_counts": dict(sorted(family_counts.items())),
            "artifacts": [store.inspect(reference) for reference in result.manifest.outputs],
        }
    )
    return validation


def _load_records(path: Path, *, wrapper: str) -> Sequence[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        values: Any = _read_jsonl(path)
    else:
        with path.open(encoding="utf-8") as handle:
            values = json.load(handle)
    if isinstance(values, Mapping):
        values = values.get(wrapper, [values])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{path} must contain a sequence of {wrapper}")
    if any(not isinstance(value, Mapping) for value in values):
        raise ValueError(f"{path} must contain only JSON objects")
    return values


def _read_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
    return values


def _input_reference(path: Path) -> ArtifactRef:
    if not path.is_file():
        raise FileNotFoundError(path)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # The digest carries input identity. An absolute workstation path would
    # make an otherwise identical manifest differ across machines.
    return ArtifactRef(path=path.name, sha256=sha256_file(path), media_type=media_type)


def _require_unique(values: Iterable[str], label: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {', '.join(duplicates)}")


def _validate_group_leakage(tasks: Sequence[Task], group_by: str) -> None:
    memberships: dict[str, set[Split]] = defaultdict(set)
    if group_by == "rule_family":
        for task in tasks:
            if task.split is not None:
                group = task.rule_family or "|".join(sorted(task.knowledge_unit_ids))
                memberships[group].add(task.split)
    elif group_by == "knowledge_unit":
        for task in tasks:
            if task.split is not None:
                for unit_id in task.knowledge_unit_ids:
                    memberships[unit_id].add(task.split)
    else:
        raise ValueError(f"unknown group_by value in manifest: {group_by!r}")
    leaking = sorted(group for group, splits in memberships.items() if len(splits) > 1)
    if leaking:
        raise ValueError("structural split leakage detected for: " + ", ".join(leaking))


__all__ = [
    "BuildResult",
    "compile_build",
    "inspect_build",
    "load_build",
    "load_canonical_units",
    "validate_build",
]
