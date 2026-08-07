#!/usr/bin/env bash
set -euo pipefail

# Offline, bounded completion gate for Prime Agent autonomous runs.
# It proves the local source-to-artifact contract only.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

build_dir="$(mktemp -d /tmp/source2agent-gate.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT

python -m pytest
python -m ruff check .

python -m volume2gym compile \
  --volume-id lantern-ledger-demo \
  --units examples/lantern_ledger/units.json \
  --output "$build_dir/build" \
  --seed 7

python -m volume2gym validate "$build_dir/build"
python -m volume2gym inspect-artifacts "$build_dir/build"
python -m volume2gym reference-eval "$build_dir/build" \
  --output "$build_dir/reference-eval" \
  --split test

test -s "$build_dir/build/manifest.json"
test -s "$build_dir/reference-eval/evaluation.json"

echo "Source2Agent local quality gate: PASS"
