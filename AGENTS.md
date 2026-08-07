# Source2Agent agent instructions

## Mission

This repository is being rebranded from **Volume2Gym** to **Source2Agent**.

Source2Agent compiles an authoritative source into a reproducible, source-grounded agent environment:

```
source volume → cited knowledge units → taskset → verifier → holdouts
             → trainer export → model evaluation → failure ledger → deployment
```

The Python package remains `volume2gym` during the compatibility phase. Do not rename the import package, module paths, or existing `v2g` command unless a migration test and compatibility alias are added first.

## Prime Agent operating rules

1. Read this file and the relevant files under `docs/` before editing.
2. Work in a branch. Do not push directly to `main`.
3. Start with the smallest executable vertical slice.
4. Run `bash scripts/prime-agent-gate.sh` after every material change.
5. Never claim neural training, generalization, deployment, or model improvement unless a stored artifact and reproducible command prove it.
6. Preserve source citations, source revisions, semantic holdouts, reward ledgers, and SHA-256 artifact references.
7. Keep the local path offline-first. Paid training, model downloads, and hosted deployment require an explicit user decision and credentials.
8. Do not add credentials, tokens, private source scans, or generated model weights to Git.
9. Do not turn the railroad proof case into the product identity. Railroad is the first useful proof volume; the thesis is general source-to-agent compilation.
10. Do not reintroduce Dakota or other community-language projects into this flagship repository. Those remain separate, ethically bounded case studies.
11. Treat Entire as the coding-agent trace layer. Prime Agent is Pi-based, so use Entire's `pi` integration when tracing Prime Agent sessions; do not invent a `prime-agent` adapter.
12. Do not push prompts, transcripts, or checkpoint metadata to the public code branch without an explicit privacy decision. Entire redaction is best-effort; keep credentials and private source material out of agent context.

## Definition of done for the flagship

The project is ready for a first public release when all of these are true:

- `python -m pytest` passes.
- `python -m ruff check .` passes.
- The Lantern & Ledger fixture compiles, validates, exports, and completes symbolic reference evaluation.
- The HTTP deployment smoke test returns `healthy` and compiles the fixture.
- The README explains the product, the compatibility boundary, and the evidence limits.
- A Prime Agent run can execute the quality gate without network access.
- An Entire session/checkpoint can be linked to the corresponding commit/PR and gate result, without weakening the code/evaluation evidence boundary.
- The full-book railroad run has a separate, explicit evidence record; it is not implied by the fixture.
- A Prime Lab / verifiers.v1 environment is added only after its exact taskset, harness, runtime, and reward contract are implemented and tested.

## Canonical commands

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .

python -m volume2gym compile \
  --volume-id lantern-ledger-demo \
  --units examples/lantern_ledger/units.json \
  --output build/lantern-ledger \
  --seed 7

python -m volume2gym validate build/lantern-ledger
python -m volume2gym inspect-artifacts build/lantern-ledger
python -m volume2gym reference-eval build/lantern-ledger \
  --output build/lantern-ledger/reference-eval \
  --split test
```

## Prime Agent delegation pattern

Use subagents for bounded, reviewable work:

- `compiler-auditor`: inspect source contracts, split invariants, and artifact hashes.
- `deployment-engineer`: run the local HTTP/Docker smoke test and improve only deployment files.
- `prime-integration`: map the current artifacts to Prime Intellect verifiers.v1 without inventing an environment ID.
- `evidence-editor`: update README and model/evaluation claims from existing artifacts only.
- `trace-auditor`: verify Entire session/checkpoint coverage and privacy before a traced agent run is pushed.

Every subagent must report changed files, commands run, failures, and remaining uncertainty. Merge no subagent output without inspecting the diff and rerunning the gate.

## Trace integration

See `docs/integrations/entire.md`. The intended setup for a Prime Agent worktree is:

```bash
entire enable --agent pi
entire status
```

Entire explains how a coding agent reached a commit; Source2Agent artifacts and the quality gate prove what the repository does. Keep those claims separate.
