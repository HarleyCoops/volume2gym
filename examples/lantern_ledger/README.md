# Lantern & Ledger

This is a tiny, fictional tabletop protocol written specifically for
Volume2Gym. It is not adapted from a published book and contains no railroad
material. The text is released under this repository's Apache-2.0 license.

The three canonical knowledge units deliberately use different rule families.
Compiling them proves that the generic path can ingest a non-railroad volume,
generate all six task families, and keep semantic groups out of one another's
train, development, and test splits.

From the repository root:

```bash
python -m pip install -e .
python -m volume2gym compile \
  --volume-id lantern-ledger-demo \
  --units examples/lantern_ledger/units.json \
  --output build/lantern-ledger \
  --seed 7
python -m volume2gym validate build/lantern-ledger
python -m volume2gym inspect-artifacts build/lantern-ledger
python -m volume2gym export build/lantern-ledger \
  --format sft \
  --split train \
  --output build/lantern-ledger/sft-train.jsonl
python -m volume2gym export build/lantern-ledger \
  --format grpo \
  --split train \
  --output build/lantern-ledger/grpo-train.jsonl
python -m volume2gym reference-eval build/lantern-ledger \
  --split test \
  --output build/lantern-ledger/reference-eval
```

The current compiler emits 3 knowledge units and 18 tasks: one task per source
unit in each of the six task families. With three semantic groups, the default
splitter assigns one complete family group—six tasks—to each split.

The reference evaluation uses the answer key as a transparent symbolic policy.
It is deliberately non-neural: the example demonstrates the compiler and
artifact loop, not a trained model or measured cross-volume generalization.
