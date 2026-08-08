#!/usr/bin/env python3
"""Extract the independently reproducible railroad-1959-v0 rule corpus.

Input is reading-order text produced by ``pdftotext -raw`` from an OCRed copy of
the pinned scan. The original scan and OCR PDF stay outside this repository
because their redistribution status has not been established.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOCUMENT_ID = "railroad-1959"
VOLUME_ID = "railroad-1959-v0"
NUMERIC_START = re.compile(
    r"^(?P<id>(?:(?:S|D|SD|ABS|ACS|CTC|IBS|MBS|TWC|YARD)[- ]?)?"
    r"\d{1,4}(?:\s*\([A-Z0-9]+\))?)(?P<punct>[.,])\s*(?P<text>.*)$"
)
GENERAL_START = re.compile(r"^(?P<id>[A-HJ-M])\.\s*(?P<text>.*)$")

# OCR rendered the terminal period as a comma at these unambiguous rule starts.
COMMA_CORRECTIONS = {
    (8, "4"),
    (12, "14"),
    (17, "S-20"),
    (19, "34"),
    (25, "99"),
    (39, "223"),
    (40, "224"),
    (81, "314"),
    (83, "334"),
    (85, "364"),
    (85, "371"),
    (90, "514"),
    (91, "612"),
    (93, "637"),
    (110, "924"),
}

# Table references and communicating-signal codes that match rule syntax.
EXCLUDED_PERIOD_CANDIDATES = {
    (12, "99"),
    (48, "99"),
    (62, "221"),
    (81, "5"),
}

SECTION_STARTS = (
    (4, "General Rules"),
    (8, "Standard Time and Timetables"),
    (10, "Signals"),
    (21, "Superiority and Movement of Trains"),
    (33, "Train Order Signals"),
    (34, "Movement by Train Orders"),
    (41, "Forms of Train Orders"),
    (51, "Signal Systems"),
    (77, "General Signal Rules"),
    (78, "Centralized Traffic Control"),
    (79, "Dual Control Switches"),
    (80, "Electric Locked Switches"),
    (81, "Manual Block System"),
    (87, "Railroad Radio Rules"),
    (89, "Automatic Block Signal System"),
    (92, "Interlocking Rules"),
    (95, "Additional General Rules"),
    (98, "Accidents and Injuries"),
    (100, "Fire and Explosives"),
    (101, "Train and Yard Service"),
    (106, "Passenger Service"),
    (109, "Freight Service"),
    (111, "Engine Service and Station Agents"),
    (114, "Train Dispatchers"),
    (115, "Legal Proceedings and Accidents"),
)

REMOVABLE_HEADINGS = {section.upper() for _, section in SECTION_STARTS} | {
    "OPERATING RULES",
    "STANDARD TIME",
    "TIME-TABLES",
    "COLOR SIGNALS",
    "HAND, FLAG AND LANTERN SIGNALS",
    "ENGINE WHISTLE SIGNALS",
    "COMMUNICATING SIGNALS",
}


@dataclass(frozen=True)
class Start:
    page: int
    line: int
    rule_id: str
    initial_text: str
    corrected: bool = False


def section_for(page: int, rule_id: str = "") -> str:
    if rule_id.isdigit() and int(rule_id) >= 1251:
        return "Legal Proceedings and Accidents"
    return next(section for start, section in reversed(SECTION_STARTS) if page >= start)


def normalize_rule_id(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"^(S|D|SD|ABS|ACS|CTC|IBS|MBS|TWC|YARD)\s+", r"\1-", value)
    return value.replace(" ", "")


def candidate_starts(pages: list[str]) -> tuple[list[Start], list[dict[str, Any]]]:
    starts: list[Start] = []
    excluded: list[dict[str, Any]] = []

    for index, line in enumerate(pages[4].splitlines()):
        match = GENERAL_START.match(line.strip())
        if match:
            starts.append(Start(4, index, match["id"], match["text"]))

    for page in range(8, min(116, len(pages))):
        for index, line in enumerate(pages[page].splitlines()):
            match = NUMERIC_START.match(line.strip())
            if not match:
                continue
            rule_id = normalize_rule_id(match["id"])
            key = (page, rule_id)
            punctuation = match["punct"]
            if punctuation == "," and key not in COMMA_CORRECTIONS:
                continue
            if punctuation == "." and key in EXCLUDED_PERIOD_CANDIDATES:
                excluded.append(
                    {"page_number": page, "rule_id": rule_id, "line": line.strip()}
                )
                continue
            starts.append(
                Start(page, index, rule_id, match["text"], punctuation == ",")
            )
    return sorted(starts, key=lambda item: (item.page, item.line)), excluded


def _is_application_heading(line: str) -> bool:
    compact = line.strip().upper()
    return compact.startswith(("APPLY ONLY", "APPLIES ONLY", "THE FOLLOWING RULE"))


def _clean_lines(lines: list[str]) -> str:
    filtered: list[str] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", raw.strip())
        if line.upper() == "TABLE OF CONTENTS":
            break
        if not line or re.fullmatch(r"\d{1,4}", line):
            continue
        if line.upper() in REMOVABLE_HEADINGS or _is_application_heading(line):
            continue
        filtered.append(line)

    joined: list[str] = []
    for line in filtered:
        if joined and joined[-1].endswith("-") and line[:1].islower():
            joined[-1] = joined[-1][:-1] + line
        else:
            joined.append(line)
    return " ".join(joined).strip()


def _application_context(pages: list[str], start: Start) -> str | None:
    lines = pages[start.page].splitlines()
    prior = [
        re.sub(r"\s+", " ", line.strip())
        for line in lines[max(0, start.line - 8) : start.line]
    ]
    selected: list[str] = []
    collecting = False
    for line in prior:
        if _is_application_heading(line):
            collecting = True
        if collecting and line and not re.fullmatch(r"\d{1,3}", line):
            selected.append(line)
    return " ".join(selected) or None


def _span_lines(pages: list[str], start: Start, end: Start | None) -> list[str]:
    final_page = end.page if end else 115
    result = [start.initial_text]
    for page in range(start.page, min(final_page + 1, len(pages))):
        lines = pages[page].splitlines()
        first = start.line + 1 if page == start.page else 0
        last = end.line if end and page == end.page else len(lines)
        result.extend(lines[first:last])
    return result


def extract_rules(pages: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    starts, excluded = candidate_starts(pages)
    raw_counts = Counter(start.rule_id for start in starts)
    occurrences: Counter[str] = Counter()
    rules: list[dict[str, Any]] = []

    for position, start in enumerate(starts):
        occurrences[start.rule_id] += 1
        occurrence = occurrences[start.rule_id]
        rule_id = start.rule_id
        if raw_counts[start.rule_id] > 1 and occurrence > 1:
            rule_id = f"{start.rule_id}@p{start.page:03d}-v{occurrence}"
        next_start = starts[position + 1] if position + 1 < len(starts) else None
        text = _clean_lines(_span_lines(pages, start, next_start))
        if not text:
            raise ValueError(f"empty extracted text for Rule {rule_id} on PDF page {start.page}")
        confidence = 0.68 if 51 <= start.page <= 76 else 0.84
        if start.corrected:
            confidence -= 0.08
        context = _application_context(pages, start)
        rules.append(
            {
                "rule_id": rule_id,
                "title": f"Rule {start.rule_id}"
                + (f" (variant {occurrence})" if occurrence > 1 else ""),
                "text": text,
                "category": section_for(start.page, start.rule_id),
                "section": section_for(start.page, start.rule_id),
                "page_number": start.page,
                "confidence": round(confidence, 2),
                "conditions": [context] if context else [],
                "required_actions": [],
                "forbidden_actions": [],
                "procedure_steps": [],
                "exceptions": [],
                "metadata": {
                    "document_id": DOCUMENT_ID,
                    "original_rule_id": start.rule_id,
                    "occurrence": occurrence,
                    "ocr_review_status": "unreviewed",
                    "ocr_punctuation_corrected": start.corrected,
                    "pdf_page_number": start.page,
                },
            }
        )

    duplicate_ids = {rule_id: count for rule_id, count in sorted(raw_counts.items()) if count > 1}
    report = {
        "volume_id": VOLUME_ID,
        "document_id": DOCUMENT_ID,
        "extracted_rule_count": len(rules),
        "compiled_task_count_expected": len(rules) * 6,
        "task_families_per_rule": 6,
        "section_counts": dict(sorted(Counter(rule["section"] for rule in rules).items())),
        "duplicate_original_rule_ids": duplicate_ids,
        "punctuation_corrections": [
            {"page_number": start.page, "rule_id": start.rule_id, "from": ",", "to": "."}
            for start in starts
            if start.corrected
        ],
        "excluded_rule_like_candidates": excluded,
    }
    return rules, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-text", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rules-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()

    pages = args.ocr_text.read_text(encoding="utf-8", errors="replace").split("\f")
    if len(pages) - 1 != 117:
        raise ValueError(f"expected 117 PDF pages, found {len(pages) - 1}")
    rules, report = extract_rules(pages)
    for rule in rules:
        rule["metadata"].update(
            {
                "source_revision": args.source_revision,
                "source_sha256": args.source_sha256,
                "source_path": "Public/1959RailRoadCodeRL.pdf",
            }
        )
    payload = {
        "volume_id": VOLUME_ID,
        "document": {
            "document_id": DOCUMENT_ID,
            "title": "Joint Form 1 Consolidated Code of Operating Rules—Revised 1959",
            "source_repository": "HarleyCoops/Qwen3-RailroadEngineer1959-RL",
            "source_path": "Public/1959RailRoadCodeRL.pdf",
            "source_revision": args.source_revision,
            "source_sha256": args.source_sha256,
            "page_count": 117,
            "rights_status": "unknown",
            "redistribution_cleared": False,
        },
        "rules": rules,
    }
    args.rules_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.rules_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report.update(
        {
            "source_revision": args.source_revision,
            "source_sha256": args.source_sha256,
            "ocr_text_sha256": _sha256(args.ocr_text),
            "source_page_count": 117,
            "ocr_pipeline": (
                "ocrmypdf --force-ocr --rotate-pages --deskew --optimize 1; "
                "pdftotext -raw"
            ),
            "ocr_review_status": "unreviewed",
            "rights_status": "unknown",
            "redistribution_cleared": False,
        }
    )
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
