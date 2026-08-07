# Source-to-agent architecture

Source2Agent is a compiler and evidence contract. It is not a chatbot wrapper.

## Product boundary

| Layer | Responsibility | Evidence |
|---|---|---|
| Source | Preserve the volume, revision, rights, and citations | source manifest |
| Knowledge | Convert source spans into stable knowledge units | unit schema + citations |
| Taskset | Generate ordinary, edge, conflict, exception, violation, and distractor tasks | canonical JSONL |
| Verifier | Score answers with inspectable components and safety gates | reward ledger |
| Holdout | Keep connected semantic groups out of other splits | split manifest |
| Trainer | Export SFT/GRPO-ready records or connect to a hosted trainer | recipe + run manifest |
| Agent | Use the taskset, tools, runtime, and model to solve tasks | rollout trace |
| Evaluation | Compare base, adapted, symbolic, and human-reviewed results | evaluation report |
| Deployment | Expose the checked artifact behind a reproducible service | image, endpoint, smoke result |

## What already works

The current package implements the source contract, deterministic compiler, grouped splits, artifact hashes, single-turn gym, reward ledger, trainer exports, symbolic reference policy, and failure curriculum shape.

The Lantern & Ledger fixture is intentionally small and rights-safe. It proves the local contract. It does not prove neural learning, railroad full-book coverage, or cross-volume generalization.

## The next research proof

The railroad lineage is the first useful proof volume, not the product identity. The required evidence is:

1. compile the full volume from a pinned source revision;
2. record the number and provenance of extracted units;
3. create semantic-group train/dev/test holdouts;
4. compare a symbolic reference, base model, SFT/LoRA model, and GRPO model;
5. publish per-family reward ledgers and failure clusters;
6. rerun the same compiler on a second rights-safe volume;
7. deploy only the checked adapter and expose its evidence limits.

## Prime Intellect mapping

Prime Intellect's current environment model maps cleanly:

- Source2Agent artifacts become the **taskset**.
- The model-facing solver becomes the **harness**.
- The local server, Docker container, or Prime Sandbox becomes the **runtime**.
- The deterministic verifier and reward ledger become the **rubric**.
- Prime Agent is the coding/research harness used to evolve the implementation; it is not conflated with the trained task-solving model.

This distinction must remain visible in code, documentation, model cards, and public claims.
