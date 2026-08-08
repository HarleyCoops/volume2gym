# railroad-1959-v1 evidence

This directory contains the deterministic, model-free evaluation record for the
local Prime Lab / `verifiers.v1` environment.

| Artifact | Contents |
|---|---|
| `summary.json` | Contract, counts, hashes, explicit limits, and non-neural status |
| `gold-validation-summary.json` | Released Verifiers gold-validator outcome for all 2,742 tasks |
| `traces.jsonl.gz` | 276 deterministic verifiers v1 test traces |
| `reward-ledgers.jsonl.gz` | 276 complete Source2Agent reward ledgers |

Reproduce byte-for-byte from the repository root:

```bash
PYTHONPATH=src:environments/railroad_1959_v1 \
  python scripts/evaluate_prime_railroad_v1.py --check
```

The policy is `symbolic-answer-key`. No provider endpoint, model inference,
training, hosted runtime, or public checkpoint was used. The 1.0 score is a
contract-consistency result only. Held-out prompts include their source excerpt,
so the split tests in-context source following rather than memorized or
cross-volume knowledge generalization.
The traces were constructed and scored locally; they do not claim execution of
a model-facing harness or rollout runtime.
