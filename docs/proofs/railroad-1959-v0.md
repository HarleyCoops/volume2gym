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

## Prime Lab / verifiers.v1 vertical slice

The local environment package at
`environments/railroad_1959_v1` targets the released `verifiers==0.3.0` API
(release source commit `0a4d872f021022310a08ec213a25f4efb4a0244a`). Its exact
taskset is a deterministic gzip artifact generated from the build above:

- local environment ID: `railroad-1959-v1`;
- Environment Hub ID: none assigned or claimed;
- taskset: 2,742 tasks, SHA-256
  `4013a7ba46365657103071c146bde0359fea548e7a317bcaddc114a3c12c6ccf`
  after decompression;
- holdout unit: complete knowledge unit;
- group counts: 365 train, 46 development, 46 test;
- pairwise group intersections: zero;
- harness: built-in `null`, one model turn;
- local runtime: subprocess, with no network-isolation claim;
- reward: `volume2gym.deterministic-composite` version 1 with its five component
  weights and safety hard gate.

The released Verifiers gold validator accepted all 2,742 tasks with zero errors,
invalid rows, missing rows, or timeouts. The committed model-free evaluation has
276 v1 traces and 276 full reward ledgers. Its symbolic answer-key policy scores
1.0, and `neural_model` is explicitly false.

Those symbolic traces were constructed and scored locally; they did not execute
a model-facing harness or rollout runtime. The subprocess runtime is an exact
configuration contract validated by dry-run, not evidence of model inference.

This slice also records a material reward limitation: all 2,742 OCR-derived
tasks have empty `required_actions`, `forbidden_actions`, `procedure_order`, and
`terms` answer-key fields. Under verifier v1, only applicable-rule citation
varies. Reference-answer fidelity is retained as an unscored metric. The perfect
symbolic result therefore proves API, taskset, trace, and reward-contract
consistency; it does not prove a strong reasoning rubric, inference quality, or
learning. Each held-out prompt contains its source excerpt, so the structural
holdout tests in-context source following rather than memorized or cross-volume
knowledge generalization.

Reproduce it locally without model inference or hosted credits:

```bash
python scripts/build_prime_railroad_v1.py --check
python scripts/evaluate_prime_railroad_v1.py --check
validate @ configs/prime/railroad-1959-v1.validate.toml --only-gold True
```

No hosted evaluation was run, no checkpoint was published, and no Entire trace
was claimed. Entire remains configured conceptually as the Pi coding-agent trace
layer; this execution environment did not have the Entire CLI, and public
checkpoint privacy has not been approved.

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
