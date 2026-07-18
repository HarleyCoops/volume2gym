# Volume2Gym Railroad 1959 — Symbolic Rule 99 Adapter

> **This is transparent and non-neural.** It is a one-rule reference artifact,
> not a trained checkpoint and not evidence of entire-book learning.

This adapter closes the smallest auditable
[Volume2Gym](https://github.com/HarleyCoops/volume2gym) fixture loop. Its JSON
files encode the source-grounded conditions, required and forbidden actions,
procedure, and terminology used to construct a structured reference answer for
the Rule 99 contract dataset.

## Connected artifacts

| Artifact | Role | Link |
|---|---|---|
| Volume2Gym | Canonical generic implementation | [GitHub](https://github.com/HarleyCoops/volume2gym) |
| Rule 99 dataset | Six train rows, one held-out row, ledgers, and curriculum artifacts | [Dataset card](https://huggingface.co/datasets/HarleyCooper/volume2gym-railroad-1959) |
| Historical railroad source | Entire-book predecessor lineage | [GitHub](https://github.com/HarleyCoops/Qwen3-RailroadEngineer1959-RL/tree/main/RailroadEngineer1959) |
| Qwen3-4B railroad LoRA | Separate neural artifact | [Model card](https://huggingface.co/HarleyCooper/Qwen3-4B-RailRoadEngineer1959) |

## Contents

- `adapter_config.json`: artifact type, source contract, training-example count,
  and structured output fields;
- `adapter_weights.json`: inspectable rule, condition, action, procedure, and
  terminology entries; and
- `adapted_outputs.jsonl`: the adapter output for the held-out fixture.

## Recorded fixture result

- Baseline mean score: `0.00`
- Symbolic adapter mean score: `1.00`
- Evaluated held-out records: `1`

This result is a one-item contract self-test. It does not measure neural
learning, cross-rule transfer, the reported full railroad corpus, or
generalization to another volume.

## Project scope

Volume2Gym is the generalized method. The entire 1959 railroad book is the
first reported full-volume implementation lineage. Rule 99 is only the complete
scope of this small regression fixture.

The executable compiler, deterministic verifier, structural splitting, SFT and
GRPO exports, symbolic reference evaluation, and a rights-safe non-railroad
example are in the
[Volume2Gym repository](https://github.com/HarleyCoops/volume2gym).

## Safety

This research artifact is not a certified operating assistant and is not a
substitute for current railroad rules, qualified personnel, or applicable law.

