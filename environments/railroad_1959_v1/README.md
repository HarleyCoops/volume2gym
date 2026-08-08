# railroad-1959-v1

This is the local Prime Lab / `verifiers.v1` taskset for Source2Agent's
`railroad-1959-v0` proof build. It targets the released `verifiers==0.3.0` API.

The packaged taskset contains exactly 2,742 tasks from Source2Agent build
`railroad-1959-v0-f5ff30c07de405de`. All six tasks generated from one knowledge
unit remain in the same split. The package verifies both its compressed and
uncompressed taskset hashes before loading.

Install from the repository root:

```bash
uv sync --locked --project environments/railroad_1959_v1 --group dev
```

The exact transitive environment and local quality-gate tools are recorded in
`uv.lock`; CI installs that lock with `uv sync --locked`.

Run the model-free gold validation without spending hosted credits:

```bash
validate @ configs/prime/railroad-1959-v1.validate.toml --only-gold True
```

The evaluation configuration pins the `null` single-turn harness and local
subprocess runtime. Subprocess is a debugging runtime and cannot enforce a
network policy; use an isolated Docker or Prime runtime before a model-backed
production evaluation. The config deliberately sets `push = false`. Override
its placeholder model only after choosing and approving a model-backed run.
The committed symbolic traces are constructed and scored locally and do not
claim that this model-facing harness/runtime path was executed.

The reward is Source2Agent deterministic verifier v1: five weighted components
plus a safety hard gate. In this OCR v0 corpus, every extracted action,
prohibition, procedure, and terminology field is empty. Consequently, only the
applicable-rule component varies. Reference-answer fidelity is recorded as an
unscored metric. This is a known evidence limit, not proof of a strong railroad
reasoning reward. Every held-out prompt contains its source excerpt, so these
holdouts test in-context source following rather than memorized or cross-volume
knowledge generalization.
