# Entire tracing integration

Entire is the trace and provenance layer for Source2Agent coding-agent work. It records the path from prompt to agent actions to files to commit, while keeping session metadata on the separate `entire/checkpoints/v1` branch.

Prime Agent is Pi-based, so use Entire's Pi integration for Prime Agent sessions:

```bash
# Install Entire using the official installer, then authenticate if required.
curl -fsSL https://entire.io/install.sh | bash

# From a clean Source2Agent worktree:
entire enable --agent pi
entire status
```

For a public repository, decide where checkpoint metadata is allowed to live before pushing:

- Entire can keep checkpoint data in the repository's `entire/checkpoints/v1` branch.
- Session metadata may include prompts, transcripts, files touched, tool calls, and token usage.
- Secret redaction is best-effort; do not place credentials or private source material in agent prompts.
- If traces must remain private, configure a private checkpoint remote or use a private development mirror. Do not assume the public Source2Agent repo is an acceptable trace sink.
- `entire enable --local --agent pi` keeps local settings out of the shared project settings; it does not by itself make checkpoint data private.

## Source2Agent evidence loop

A completed agent run should be linkable across three layers:

1. Entire session/checkpoint: why the agent acted and what it touched;
2. Git commit/PR: the reviewed source change;
3. Source2Agent gate/deployment artifacts: whether the change actually works.

The quality gate remains authoritative for repository correctness:

```bash
bash scripts/prime-agent-gate.sh
entire status
```

Entire evidence does not replace tests, verifiers, holdout evaluation, or deployment checks. It explains the engineering trajectory around them.

## Prime Agent launch

After Entire is enabled in the worktree, launch Prime Agent with the repository's `AGENTS.md` and the runbook goal. Commit only after the gate passes. Entire will checkpoint the agent session alongside that commit.

For CI or headless use, do not add tokens to the repository. Use the credential mechanism appropriate to the Entire deployment and machine, then verify with `entire status`.

Authoritative references:

- Entire CLI: https://github.com/entireio/cli
- Entire setup: https://entire.io/
