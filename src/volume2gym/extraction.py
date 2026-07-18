"""Provider-neutral extraction of cited knowledge units from source spans.

The provider is responsible only for describing the knowledge in a span.  This
module owns identity and provenance: citations are built from the caller's
``SourceDocument`` and ``SourceSpan`` records and can never be supplied by a
model response.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .models import Citation, KnowledgeKind, KnowledgeUnit, SourceDocument, SourceSpan
from .providers import GenerationRequest, StructuredGenerator

EXTRACTION_SCHEMA_NAME = "volume2gym.knowledge_units.v0.1"


class ExtractionError(RuntimeError):
    """Raised when source spans or generated knowledge cannot be used safely."""


class _ExtractedUnit(BaseModel):
    """The deliberately provenance-free unit shape accepted from a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str | None = None
    kind: KnowledgeKind = KnowledgeKind.CONCEPT
    title: str
    text: str
    family: str | None = None
    section: str | None = None
    conditions: tuple[str, ...] = ()
    required_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    procedure_steps: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    related_unit_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("unit_id", "title", "text", "family", "section")
    @classmethod
    def reject_blank_scalar(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator(
        "conditions",
        "required_actions",
        "forbidden_actions",
        "procedure_steps",
        "exceptions",
        "terms",
        "related_unit_ids",
    )
    @classmethod
    def reject_blank_list_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("items must not be blank")
        return normalized


class _ExtractionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_units: tuple[_ExtractedUnit, ...] = Field(min_length=1)


class StructuredKnowledgeExtractor:
    """Extract normalized units one span at a time through any structured provider.

    Calls follow the order of the supplied spans.  The order must be strictly
    increasing by ``ordinal`` so accidental source shuffling is reported rather
    than silently changing the build.
    """

    def __init__(
        self,
        generator: StructuredGenerator,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.generator = generator
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(
        self,
        document: SourceDocument,
        spans: Sequence[SourceSpan],
    ) -> list[KnowledgeUnit]:
        checked_spans = _validate_spans(document, spans)
        units: list[KnowledgeUnit] = []
        seen_unit_ids: set[str] = set()

        for span in checked_spans:
            # Provider errors deliberately cross this boundary unchanged.  A
            # failed span must stop the build instead of looking like no rules.
            result = self.generator.generate(
                build_extraction_request(
                    document,
                    span,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
            try:
                envelope = _ExtractionEnvelope.model_validate(result.value)
            except ValidationError as exc:
                raise ExtractionError(
                    f"invalid extraction response for span {span.span_id!r}: {exc}"
                ) from exc

            citation = _citation_for(document, span)
            for extracted in envelope.knowledge_units:
                unit_id = extracted.unit_id or _stable_unit_id(document, span, extracted)
                if unit_id in seen_unit_ids:
                    raise ExtractionError(f"duplicate knowledge unit id {unit_id!r}")
                seen_unit_ids.add(unit_id)
                metadata = dict(extracted.metadata)
                metadata["extraction"] = {
                    "provider": result.provider,
                    "model": result.model,
                    "schema_name": EXTRACTION_SCHEMA_NAME,
                    "span_ordinal": span.ordinal,
                }
                units.append(
                    KnowledgeUnit(
                        unit_id=unit_id,
                        kind=extracted.kind,
                        title=extracted.title,
                        text=extracted.text,
                        family=extracted.family,
                        section=extracted.section,
                        citations=(citation,),
                        conditions=extracted.conditions,
                        required_actions=extracted.required_actions,
                        forbidden_actions=extracted.forbidden_actions,
                        procedure_steps=extracted.procedure_steps,
                        exceptions=extracted.exceptions,
                        terms=extracted.terms,
                        related_unit_ids=extracted.related_unit_ids,
                        confidence=extracted.confidence,
                        metadata=metadata,
                    )
                )
        return units


def extract_knowledge_units(
    generator: StructuredGenerator,
    document: SourceDocument,
    spans: Sequence[SourceSpan],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> list[KnowledgeUnit]:
    """Convenience wrapper around :class:`StructuredKnowledgeExtractor`."""

    return StructuredKnowledgeExtractor(
        generator,
        temperature=temperature,
        max_tokens=max_tokens,
    ).extract(document, spans)


def build_extraction_request(
    document: SourceDocument,
    span: SourceSpan,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> GenerationRequest:
    """Create the deterministic provider request for one validated source span."""

    source = {
        "document_id": document.document_id,
        "title": document.title,
        "edition": document.edition,
        "span_id": span.span_id,
        "ordinal": span.ordinal,
        "pages": [span.page_start, span.page_end],
        "section_path": list(span.section_path),
        "text": span.text,
    }
    return GenerationRequest(
        system=(
            "Extract the explicit knowledge in exactly one source span. Return JSON only, "
            "with exactly one top-level key named knowledge_units. Do not return citations, "
            "document IDs, span IDs, page numbers, or other provenance; the caller attaches "
            "authoritative provenance. Each unit requires a non-empty title and text. Allowed "
            "unit fields are unit_id, kind, title, text, family, section, conditions, "
            "required_actions, forbidden_actions, procedure_steps, exceptions, terms, "
            "related_unit_ids, confidence, and metadata. Use arrays for all plural fields."
        ),
        prompt="Source span:\n" + json.dumps(source, ensure_ascii=False, sort_keys=True),
        schema_name=EXTRACTION_SCHEMA_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _validate_spans(
    document: SourceDocument,
    spans: Sequence[SourceSpan],
) -> tuple[SourceSpan, ...]:
    if not document.document_id.strip():
        raise ExtractionError("source document_id must not be blank")
    checked = tuple(spans)
    if not checked:
        raise ExtractionError("at least one source span is required")

    seen_span_ids: set[str] = set()
    previous_ordinal: int | None = None
    for position, span in enumerate(checked):
        if not isinstance(span, SourceSpan):
            raise ExtractionError(f"span at position {position} is not a SourceSpan")
        if span.document_id != document.document_id:
            raise ExtractionError(
                f"span {span.span_id!r} belongs to document {span.document_id!r}, "
                f"not {document.document_id!r}"
            )
        if not span.span_id.strip():
            raise ExtractionError(f"span at position {position} has a blank span_id")
        if span.span_id in seen_span_ids:
            raise ExtractionError(f"duplicate source span id {span.span_id!r}")
        seen_span_ids.add(span.span_id)
        if not span.text.strip():
            raise ExtractionError(f"source span {span.span_id!r} has no text")
        if previous_ordinal is not None and span.ordinal <= previous_ordinal:
            raise ExtractionError("source spans must have strictly increasing ordinals")
        previous_ordinal = span.ordinal
    return checked


def _stable_unit_id(
    document: SourceDocument,
    span: SourceSpan,
    unit: _ExtractedUnit,
) -> str:
    identity = {
        "document_id": document.document_id,
        "span_id": span.span_id,
        "kind": unit.kind.value,
        "title": unit.title,
        "text": unit.text,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "ku-" + hashlib.sha256(encoded).hexdigest()[:20]


def _citation_for(document: SourceDocument, span: SourceSpan) -> Citation:
    section = " > ".join(span.section_path) or None
    locator_parts = [f"ordinal {span.ordinal}"]
    if span.page_start is not None and span.page_end is not None:
        if span.page_start == span.page_end:
            locator_parts.append(f"page {span.page_start}")
        else:
            locator_parts.append(f"pages {span.page_start}-{span.page_end}")
    elif span.page_start is not None:
        locator_parts.append(f"page {span.page_start}")
    if section:
        locator_parts.append(f"section {section}")
    return Citation(
        document_id=document.document_id,
        span_id=span.span_id,
        page_number=span.page_start,
        section=section,
        locator="; ".join(locator_parts),
        quote=span.text,
    )


__all__ = [
    "EXTRACTION_SCHEMA_NAME",
    "ExtractionError",
    "StructuredKnowledgeExtractor",
    "build_extraction_request",
    "extract_knowledge_units",
]
