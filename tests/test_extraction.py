from __future__ import annotations

import json

import pytest

from volume2gym.extraction import (
    EXTRACTION_SCHEMA_NAME,
    ExtractionError,
    StructuredKnowledgeExtractor,
    extract_knowledge_units,
)
from volume2gym.models import KnowledgeKind, SourceDocument, SourceSpan
from volume2gym.providers import CallableGenerator, ProviderError


def _document() -> SourceDocument:
    return SourceDocument(
        document_id="handbook-2025",
        title="Operations Handbook",
        edition="2025",
        media_type="application/pdf",
        sha256="a" * 64,
    )


def _span(
    span_id: str = "span-1",
    *,
    ordinal: int = 0,
    document_id: str = "handbook-2025",
    text: str = "When the alarm sounds, stop the machine before inspection.",
) -> SourceSpan:
    return SourceSpan(
        span_id=span_id,
        document_id=document_id,
        ordinal=ordinal,
        text=text,
        page_start=7 + ordinal,
        page_end=7 + ordinal,
        section_path=("Safety", "Alarms"),
    )


def test_extracts_complete_units_and_builds_authoritative_citations() -> None:
    requests = []

    def generate(request):
        requests.append(request)
        return {
            "knowledge_units": [
                {
                    "unit_id": "alarm-stop",
                    "kind": "procedure",
                    "title": "Alarm stop",
                    "text": "Stop the machine before inspection when an alarm sounds.",
                    "family": "machine-safety",
                    "section": "Alarm response",
                    "conditions": ["The alarm sounds"],
                    "required_actions": ["Stop the machine", "Inspect after stopping"],
                    "forbidden_actions": ["Inspect while the machine is moving"],
                    "procedure_steps": ["Stop the machine", "Inspect the machine"],
                    "exceptions": ["A remote diagnostic does not require physical access"],
                    "terms": ["alarm", "stop"],
                    "related_unit_ids": ["lockout"],
                    "confidence": 0.9,
                    "metadata": {"heading_level": 2, "extraction": "untrusted"},
                }
            ]
        }

    units = StructuredKnowledgeExtractor(
        CallableGenerator(generate, provider="fixture", model="extractor-v1")
    ).extract(_document(), [_span()])

    assert len(units) == 1
    unit = units[0]
    assert unit.unit_id == "alarm-stop"
    assert unit.kind is KnowledgeKind.PROCEDURE
    assert unit.conditions == ("The alarm sounds",)
    assert unit.required_actions == ("Stop the machine", "Inspect after stopping")
    assert unit.forbidden_actions == ("Inspect while the machine is moving",)
    assert unit.procedure_steps == ("Stop the machine", "Inspect the machine")
    assert unit.exceptions == ("A remote diagnostic does not require physical access",)
    assert unit.terms == ("alarm", "stop")
    assert unit.related_unit_ids == ("lockout",)
    assert unit.metadata["heading_level"] == 2
    assert unit.metadata["extraction"] == {
        "provider": "fixture",
        "model": "extractor-v1",
        "schema_name": EXTRACTION_SCHEMA_NAME,
        "span_ordinal": 0,
    }

    citation = unit.citations[0]
    assert citation.document_id == "handbook-2025"
    assert citation.span_id == "span-1"
    assert citation.page_number == 7
    assert citation.section == "Safety > Alarms"
    assert citation.locator == "ordinal 0; page 7; section Safety > Alarms"
    assert citation.quote == _span().text
    assert citation.quote_hash is not None

    assert requests[0].schema_name == EXTRACTION_SCHEMA_NAME
    source = json.loads(requests[0].prompt.removeprefix("Source span:\n"))
    assert source["document_id"] == "handbook-2025"
    assert source["span_id"] == "span-1"
    assert source["text"] == _span().text


def test_omitted_ids_are_stable_and_calls_follow_span_order() -> None:
    requested_span_ids: list[str] = []

    def generate(request):
        source = json.loads(request.prompt.removeprefix("Source span:\n"))
        requested_span_ids.append(source["span_id"])
        return {
            "knowledge_units": [
                {
                    "title": f"Unit from {source['span_id']}",
                    "text": source["text"],
                }
            ]
        }

    spans = [_span("span-a", ordinal=0), _span("span-b", ordinal=1)]
    generator = CallableGenerator(generate)
    first = extract_knowledge_units(generator, _document(), spans)
    second = extract_knowledge_units(generator, _document(), spans)

    assert [unit.unit_id for unit in first] == [unit.unit_id for unit in second]
    assert all(unit.unit_id.startswith("ku-") for unit in first)
    assert requested_span_ids == ["span-a", "span-b", "span-a", "span-b"]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"knowledge_units": []},
        {"knowledge_units": [{"title": " ", "text": "content"}]},
        {"knowledge_units": [{"title": "Title", "text": ""}]},
        {
            "knowledge_units": [{"title": "Title", "text": "content"}],
            "ignored": True,
        },
        {
            "knowledge_units": [
                {
                    "title": "Title",
                    "text": "content",
                    "citations": [{"document_id": "invented"}],
                }
            ]
        },
    ],
)
def test_rejects_nonconforming_or_empty_provider_units(response) -> None:
    extractor = StructuredKnowledgeExtractor(CallableGenerator(lambda _request: response))

    with pytest.raises(ExtractionError, match="invalid extraction response.*span-1"):
        extractor.extract(_document(), [_span()])


def test_rejects_duplicate_unit_ids_across_spans() -> None:
    generator = CallableGenerator(
        lambda _request: {
            "knowledge_units": [
                {"unit_id": "duplicate", "title": "Unit", "text": "Content"}
            ]
        }
    )
    spans = [_span("span-a", ordinal=0), _span("span-b", ordinal=1)]

    with pytest.raises(ExtractionError, match="duplicate knowledge unit id 'duplicate'"):
        StructuredKnowledgeExtractor(generator).extract(_document(), spans)


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        ([], "at least one source span"),
        ([_span(document_id="other")], "belongs to document"),
        ([_span(text="  ")], "has no text"),
        ([_span("same"), _span("same", ordinal=1)], "duplicate source span id"),
        (
            [_span("later", ordinal=1), _span("earlier", ordinal=0)],
            "strictly increasing ordinals",
        ),
    ],
)
def test_source_span_failures_are_explicit(spans, message) -> None:
    generator = CallableGenerator(lambda _request: {"knowledge_units": []})

    with pytest.raises(ExtractionError, match=message):
        StructuredKnowledgeExtractor(generator).extract(_document(), spans)


def test_provider_failures_are_not_converted_to_empty_results() -> None:
    def fail(_request):
        raise OSError("provider unavailable")

    generator = CallableGenerator(fail, provider="fixture", model="offline")

    with pytest.raises(ProviderError, match="fixture/offline failed: provider unavailable"):
        StructuredKnowledgeExtractor(generator).extract(_document(), [_span()])
