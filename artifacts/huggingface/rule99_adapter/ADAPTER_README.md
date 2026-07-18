# Volume2Gym Symbolic Adapter

Model ID: `volume2gym-symbolic-adapter-v0`

This adapter is a transparent rule-weight artifact generated from the
Volume2Gym proof-cycle training tasks and extracted source rule schema. It is
not a neural checkpoint; it is the smallest auditable adapter that closes the
proof loop without using fixture answers at inference time.

## Contents

- `adapter_config.json`: adapter metadata, trainer, source, and output contract.
- `adapter_weights.json`: rule, action, procedure, terminology, and condition weights.

## Scope

The adapter covers `Rule 99` / `train_protection` for the
railroad 1959 proof case and should be evaluated with the included reward
ledgers and verifier.
