---
language: en
license: other
base_model: Qwen/Qwen3-4B-Instruct-2507
pipeline_tag: text-generation
tags:
- lora
- grpo
- reinforcement-learning
- tinker
- qwen
- qwen3
- rail-safety
- operations
- historical-rules
- deterministic-rubric
- safety-critical
preview_image: preview.jpg
---

# Qwen3-4B Railroad Engineer 1959 — LoRA Artifact

This repository publishes a Qwen3-4B-Instruct-2507 LoRA from the broader
Railroad Engineer 1959 training lineage. The ambition behind that lineage is the
same one now made executable in
[Volume2Gym](https://github.com/HarleyCoops/volume2gym):

> **Any single structured volume can become an RL gym that produces trainable
> models.**

The railroad book is a powerful first test track because it is procedural,
safety-sensitive, explicit about required and forbidden actions, and dense with
exceptions. The project aims to turn those structures into practice tasks,
verifiable rewards, structural holdouts, and failure-driven curricula—not only
retrieve passages from the book.

## Scope and connected artifacts

| Artifact | Scope | Link |
|---|---|---|
| Volume2Gym | Canonical generic compiler, gym, verifier, exports, tests, and provenance | [GitHub](https://github.com/HarleyCoops/volume2gym) |
| Historical railroad implementation | Reported entire-book extraction/training lineage | [GitHub](https://github.com/HarleyCoops/Qwen3-RailroadEngineer1959-RL/tree/main/RailroadEngineer1959) |
| This LoRA | Neural adapter in that broader railroad lineage | This repository |
| Rule 99 dataset | Separate one-rule contract fixture, not this model's full training corpus | [Dataset card](https://huggingface.co/datasets/HarleyCooper/volume2gym-railroad-1959) |
| Rule 99 symbolic adapter | Separate non-neural fixture reference | [Model card](https://huggingface.co/HarleyCooper/volume2gym-railroad-1959-adapter) |

Rule 99 is mentioned because it is the complete scope of the small auditable
fixture. It is not the project thesis, not the entire railroad-book scope, and
not a substitute for the training corpus reported for this LoRA.

## Files actually published here

- `adapter_model.safetensors` — LoRA weights, 142,076,360 bytes
- `adapter_config.json` — PEFT adapter configuration
- `README.md` — this model card
- preview and visualization media retained from the original publication

The LoRA weight is pinned in the Volume2Gym provenance manifest by SHA-256:

```text
8642e8b135c0ccb660c6a8f1727abfd774f4239c53ea2b5e5f724c3c9cd72f81
```

## Reported historical recipe

The original publication recorded the following training description:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Optimizer: Tinker GRPO with LoRA rank 32
- Reported task corpus: 2,708 railroad scenarios
- Reported reward: deterministic 50% safety, 30% procedure, 20% terminology,
  using exact-match/token-F1 components
- Reported sampling: maximum 256 tokens and temperature 0.9 during training
- Reported batching: batch size 32, group size 8, 77 batches, evaluation every
  20 batches

These are historical card claims. The public artifacts do not currently contain
the complete corpus, split membership, hashes, pinned base-model revision,
runnable railroad-specific launcher, environment lock, seeds, per-example
reward ledger, and run trace needed to reproduce that neural result exactly.

The preserved predecessor source instead contains a Claude-judge railroad
environment and incomplete training stubs. The new deterministic verifier in
Volume2Gym is a clean generic implementation; it should not be retroactively
treated as the exact historical neural-run verifier without a published
conversion and reconciliation record.

## What is established versus still open

| Claim | Evidence status |
|---|---|
| The LoRA weights and adapter configuration are public | **Established** in this repository |
| The historical railroad source and pipeline notes are public | **Established** and pinned by Volume2Gym |
| The pipeline notes report 536 rules and 2,708 scenarios | **Reported**, but the merged corpus is absent publicly |
| This exact adapter can be rebuilt byte-for-byte from public inputs | **Not established** |
| Held-out rule-family or chapter generalization | **Not yet measured publicly** |
| Cross-volume neural generalization | **Research target** |

The complete audit, pinned upstream revisions, omitted-file inventory, and
rights boundary are in the
[Volume2Gym artifact map](https://github.com/HarleyCoops/volume2gym/blob/main/artifacts/README.md)
and
[machine-readable provenance manifest](https://github.com/HarleyCoops/volume2gym/blob/main/provenance/upstreams.json).

## Intended use and cautions

This adapter is a research artifact for constrained historical railroad-rule
experiments. Deterministic literal rubrics can under-reward valid paraphrases,
and the public evidence does not establish current operational competence.

It is not a general assistant, current railroad operating instruction, a safety
certification, or a substitute for current rules, qualified personnel, and
applicable law. Keep a human expert in the loop for any real-world use.

## Acknowledgments

- Base model: Qwen team
- Training infrastructure reported by the original publication: Thinking
  Machines Tinker
- Dataset/rubric lineage: Railroad Engineer 1959 and Prime Intellect
- Generic compiler and consolidated provenance: Volume2Gym

