# railroad-1959-v0 proof

`railroad-1959-v0` is Source2Agent's first independently reproducible full-volume
proof. It is a deterministic OCR build, not a claim that a neural model learned
the book.

## Pinned source

- Repository: `HarleyCoops/Qwen3-RailroadEngineer1959-RL`
- Revision: `cd7cfd8bab3d9d9c33446c971f5df8276e5a29f4`
- Path: `Public/1959RailRoadCodeRL.pdf`
- SHA-256: `c96a60c2b20e7b34d9bd689d57b2ec5b8c71362545b889c479bdf04fe6444350`
- PDF pages: 117

The scan is image-only. The reproducible extraction path is:

```bash
ocrmypdf --force-ocr --rotate-pages --deskew --optimize 1 \
  1959RailRoadCodeRL.pdf railroad-1959-ocr.pdf
pdftotext -raw railroad-1959-ocr.pdf railroad-1959-raw.txt

python scripts/build_railroad_1959_v0.py \
  --ocr-text railroad-1959-raw.txt \
  --source-sha256 c96a60c2b20e7b34d9bd689d57b2ec5b8c71362545b889c479bdf04fe6444350 \
  --source-revision cd7cfd8bab3d9d9c33446c971f5df8276e5a29f4 \
  --rules-output volumes/railroad-1959-v0/rules.json \
  --report-output volumes/railroad-1959-v0/extraction-report.json
```

The extractor records 15 punctuation corrections, four excluded rule-like
references, and three duplicated original rule IDs. Duplicate railroad-specific
variants receive stable page-qualified IDs instead of being discarded.

## Verified build

```bash
python -m volume2gym compile \
  --volume-id railroad-1959-v0 \
  --output runs/railroad-1959-v0/build \
  --railroad-rules volumes/railroad-1959-v0/rules.json \
  --document-id railroad-1959 \
  --seed 1959 \
  --group-by knowledge_unit \
  --source-revision cd7cfd8bab3d9d9c33446c971f5df8276e5a29f4
```

The reproduced build contains 457 knowledge units and 2,742 tasks: 2,190 train,
276 development, and 276 test. Each of the six task families contains 457 tasks.
Validation passed artifact hashes, schemas, references, split membership, and
knowledge-unit leakage checks.

The train split exports as both SFT and GRPO JSONL. A symbolic answer-key
reference scored 1.0 on all 276 test tasks. This demonstrates internal compiler,
verifier, and deployment-contract consistency. It is explicitly not a neural
evaluation.

## Operational API

Run `python deploy/server.py`, then call:

```bash
curl http://127.0.0.1:8000/volumes/railroad-1959-v0
curl -X POST http://127.0.0.1:8000/volumes/railroad-1959-v0/compile
curl -X POST http://127.0.0.1:8000/volumes/railroad-1959-v0/reference-eval
```

The container is published to `ghcr.io/harleycoops/volume2gym` by the deployment
workflow. A public always-on HTTP endpoint remains hosting-provider work and must
not be implied by the container publication.

## Evidence boundary

Historical notes reported 536 rules and 2,708 scenarios, but the merged data
behind those counts is absent from the public lineage. This v0 therefore reports
only the independently reproduced 457/2,742 build. OCR is unreviewed, source and
derived-text redistribution rights remain unknown, and the historical rules are
not current operating instruction.
