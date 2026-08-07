# Prime Agent runbook

Prime Agent is the coding and long-running research harness for Source2Agent. It is not the trained model and it is not a security sandbox. It runs project commands with the permissions of the user who launches it.

The official Prime Agent release is open source and installs with the verified installer:

```bash
curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
```

After installation:

```prime-agent
/login
```

Use a disposable clone or clean worktree. Authenticate with the Prime Intellect subscription or API-key provider you intend to use. Do not place credentials in this repository.

## Start the Source2Agent build session

From the repository root:

```bash
prime-agent \
  --goal "Make Source2Agent operational from source volume to verified artifact and local deployment. Preserve the Volume2Gym compatibility boundary. Work in small commits, use the quality gate, and do not start paid training or hosted deployment without explicit approval." \
  --autonomous \
  --autonomous-gate "bash scripts/prime-agent-gate.sh" \
  --autonomous-gate-retries 3 \
  --autonomous-max-turns 24 \
  --autonomous-max-tokens 120000 \
  --autonomous-timeout-ms 3600000
```

The gate is deliberately offline. A passed gate proves only the local compiler, verifier, fixture, and deployment contract; it does not prove a trained model or hosted deployment.

## Recommended first prompt

```text
Read AGENTS.md, docs/prime-agent-runbook.md, docs/architecture/source-to-agent.md,
and the existing tests. Inspect the current diff before changing anything.

Implement the smallest missing vertical slice:
1. make the local Source2Agent API compile and symbolically evaluate the Lantern & Ledger fixture;
2. add deterministic tests and a Docker smoke test;
3. preserve package/import/CLI compatibility;
4. only then design the Prime Intellect verifiers.v1 adapter.

Do not invent a Prime Environment Hub ID. If hosted integration requires credentials,
an external environment ID, or an API contract not present in the repository, document
the exact blocker and leave a runnable local substitute.
```

## Continual-harness discipline

Use `/refine` only after a quality-gated run. Persist small lessons such as:

- a recurring failure mode and its verified fix;
- a stable deployment command;
- a source-provenance invariant;
- a Prime integration convention confirmed by current official documentation.

Do not let the harness rewrite product claims, source licenses, or ethical boundaries from a single successful rollout. Review every refinement as a normal code change.

## Prime Lab handoff

Prime Lab environments separate:

- **taskset** — source-grounded tasks and data;
- **harness** — the model-facing program that produces a rollout;
- **runtime** — local subprocess, Docker, or Prime Sandbox;
- **rubric/reward** — deterministic success criteria.

Source2Agent already owns the task compilation, structural holdouts, and reward ledger. The next integration should wrap those artifacts in the current `verifiers.v1` interfaces, then run a hosted evaluation before any RL job.

The integration is complete only when the repository contains:

- an exact environment/taskset identifier or a local equivalent;
- a pinned harness/runtime contract;
- a smoke evaluation with saved rollout and reward artifacts;
- a cost-bounded training configuration;
- an adapter/model card that states what was and was not learned;
- a deployment endpoint or documented inference target that has been checked after publication.

## Stop conditions

Stop and report instead of guessing when:

- Prime authentication is missing;
- a hosted run would spend credits;
- an environment ID or deployment endpoint is unknown;
- a test requires private source material;
- a model result cannot be reproduced from committed configuration and published artifacts;
- a proposed harness refinement weakens a verifier or makes provenance less inspectable.
