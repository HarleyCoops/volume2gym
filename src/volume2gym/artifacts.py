"""Deterministic, atomic persistence for Volume2Gym artifacts.

The artifact format is deliberately boring: canonical UTF-8 JSON and JSONL,
addressed by the SHA-256 digest of the bytes written to disk.  This module has
no dependency on a training framework or remote object store.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ArtifactRef

T = TypeVar("T", bound=BaseModel)


def _jsonable(value: Any) -> Any:
    """Convert supported values to JSON data without provider-specific hooks."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: canonical_json(item))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` into the one canonical JSON representation we hash."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def canonical_jsonl_bytes(records: Iterable[Any]) -> bytes:
    """Materialize canonical JSONL bytes, including a final newline per record."""

    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def hash_jsonl(records: Iterable[Any]) -> str:
    return sha256_bytes(canonical_jsonl_bytes(records))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ArtifactStore:
    """A path-confined local store for canonical build artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("artifact paths must be relative to the store root")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path escapes the store root") from exc
        return candidate

    def write_json(self, relative_path: str | Path, value: Any) -> ArtifactRef:
        payload = canonical_json_bytes(value)
        path = self.resolve(relative_path)
        _atomic_write(path, payload)
        return ArtifactRef(
            path=self._relative(path),
            sha256=sha256_bytes(payload),
            media_type="application/json",
            record_count=1,
        )

    def write_jsonl(self, relative_path: str | Path, records: Iterable[Any]) -> ArtifactRef:
        # Materializing once ensures one-shot iterators are neither double-read nor
        # hashed differently from the actual bytes written.
        encoded: list[bytes] = []
        for record in records:
            encoded.append(canonical_json_bytes(record) + b"\n")
        payload = b"".join(encoded)
        path = self.resolve(relative_path)
        _atomic_write(path, payload)
        return ArtifactRef(
            path=self._relative(path),
            sha256=sha256_bytes(payload),
            media_type="application/x-ndjson",
            record_count=len(encoded),
        )

    def read_json(self, relative_path: str | Path, model: type[T] | None = None) -> T | Any:
        path = self.resolve(relative_path)
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return model.model_validate(value) if model is not None else value

    def read_jsonl(
        self,
        relative_path: str | Path,
        model: type[T] | None = None,
    ) -> Iterator[T | Any]:
        path = self.resolve(relative_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc.msg}"
                    ) from exc
                yield model.model_validate(value) if model is not None else value

    def validate(self, reference: ArtifactRef) -> None:
        path = self.resolve(reference.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != reference.sha256:
            raise ValueError(
                f"artifact hash mismatch for {reference.path}: "
                f"expected {reference.sha256}, found {actual_hash}"
            )
        if reference.record_count is not None:
            actual_count = self._record_count(path, reference.media_type)
            if actual_count != reference.record_count:
                raise ValueError(
                    f"artifact record count mismatch for {reference.path}: "
                    f"expected {reference.record_count}, found {actual_count}"
                )

    def inspect(self, reference: ArtifactRef) -> dict[str, Any]:
        self.validate(reference)
        path = self.resolve(reference.path)
        return {
            "path": reference.path,
            "sha256": reference.sha256,
            "media_type": reference.media_type,
            "record_count": reference.record_count,
            "size_bytes": path.stat().st_size,
        }

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _record_count(path: Path, media_type: str) -> int:
        if media_type == "application/x-ndjson":
            with path.open(encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        return 1


__all__ = [
    "ArtifactStore",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "hash_json",
    "hash_jsonl",
    "sha256_bytes",
    "sha256_file",
]
