# Volume2Gym Railroad 1959 — Rule 99 Contract Fixture

> **Scope:** Rule 99 is the entire scope of this small dataset fixture. It is
> not the scope of Volume2Gym, the railroad book, or the broader neural-model
> lineage.

This dataset is the smallest auditable artifact in the
[Volume2Gym](https://github.com/HarleyCoops/volume2gym) project: a static fixture
showing how source-grounded knowledge can become scenario tasks, structured
answer contracts, deterministic reward ledgers, evaluation records, failure
clusters, and a proposed next curriculum.

The larger thesis is:

> **Any single structured volume can become an RL gym that produces trainable
> models.**

## Where this fixture fits

| Layer | Scope | Link |
|---|---|---|
| Volume2Gym | Generic compiler, gym, verifier, exports, tests, and learning-loop contracts | [GitHub](https://github.com/HarleyCoops/volume2gym) |
| Railroad Engineer 1959 | First reported entire-book implementation lineage | [Historical source](https://github.com/HarleyCoops/Qwen3-RailroadEngineer1959-RL/tree/main/RailroadEngineer1959) |
| This dataset | One Rule 99 contract/regression fixture | This repository |
| Symbolic adapter | Transparent, non-neural reference for this fixture | [Model card](https://huggingface.co/HarleyCooper/volume2gym-railroad-1959-adapter) |
| Qwen3-4B LoRA | Separate neural artifact in the broader railroad lineage | [Model card](https://huggingface.co/HarleyCooper/Qwen3-4B-RailRoadEngineer1959) |

Volume2Gym now contains an executable provider-neutral implementation and a
fictional non-railroad proof volume. The full historical 536-rule/2,708-task
railroad corpus reported by the predecessor pipeline is not present in this
dataset or in the pinned public source repository.

## Fixture inventory

- 6 training rows spanning standard, edge, violation, conflict, exception, and
  adversarial task families;
- 1 held-out scenario;
- structured baseline and adapted outputs;
- component reward ledgers and an evaluation report;
- failure clusters, a curriculum request, and a proposed next batch; and
- a transparent symbolic adapter contract.

## Source boundary

- Proof case: `railroad_1959`
- Fixture unit: `Rule 99` / protection of a stopped train where it may be
  overtaken
- Rule family: `train_protection`
- Recorded source page: 123
- Recorded source span: `rule-99.short-excerpt`
- Rights status: `source-not-redistributed-pending-release-review`

Only metadata, structured derivatives, and the already-published short fixture
are included. The source scan and the full extracted book are not redistributed
here.

## What the result means

The included one-item comparison records a baseline score of `0.00` and a
symbolic-adapter score of `1.00`. This is a contract reconciliation test over a
single held-out fixture item. It is **not** evidence of neural training,
cross-rule transfer, entire-book performance, or cross-volume generalization.

The generic implementation, verifier semantics, reproducible quickstart, pinned
artifact revisions, and evidence boundaries are documented in the
[Volume2Gym README](https://github.com/HarleyCoops/volume2gym#readme).

## Intended use

Use this fixture to test dataset loaders, source citations, structured answers,
reward-ledger reconciliation, failure mining, and compatibility with the
generic Volume2Gym contracts.

Do not use it as current railroad operating instruction, a safety
certification, or a substitute for qualified personnel and applicable rules.

