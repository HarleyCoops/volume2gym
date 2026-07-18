"""Source ingestion helpers with no provider or domain coupling."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SourceError(RuntimeError):
    """Raised when a source cannot be read without losing provenance."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RenderedPage:
    page_number: int
    path: Path
    media_type: str
    sha256: str


class PDFRenderer:
    """Render selected PDF pages while retaining page identity and file hashes."""

    def render(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        *,
        dpi: int = 150,
        pages: Iterable[int] | None = None,
    ) -> list[RenderedPage]:
        source = Path(pdf_path)
        target = Path(output_dir)
        if not source.is_file():
            raise FileNotFoundError(source)
        if dpi <= 0:
            raise ValueError("dpi must be greater than zero")

        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - exercised without the optional extra
            raise SourceError("PDF support requires `pip install volume2gym[pdf]`") from exc

        target.mkdir(parents=True, exist_ok=True)
        document = fitz.open(source)
        try:
            selected = list(pages) if pages is not None else list(range(1, len(document) + 1))
            invalid = [page for page in selected if page < 1 or page > len(document)]
            if invalid:
                raise ValueError(f"page numbers outside 1..{len(document)}: {invalid}")

            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            rendered: list[RenderedPage] = []
            for page_number in selected:
                output = target / f"page_{page_number:04d}.png"
                document.load_page(page_number - 1).get_pixmap(matrix=matrix).save(output)
                rendered.append(
                    RenderedPage(
                        page_number=page_number,
                        path=output,
                        media_type="image/png",
                        sha256=sha256_file(output),
                    )
                )
            return rendered
        finally:
            document.close()


def detect_media_type(path: str | Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise SourceError(f"expected an object at {path}:{line_number}")
            yield record


def iter_source_files(directory: str | Path) -> Iterator[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    yield from sorted(path for path in root.rglob("*") if path.is_file())
