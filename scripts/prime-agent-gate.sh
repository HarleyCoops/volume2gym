#!/usr/bin/env bash
set -euo pipefail

# Offline, bounded completion gate for Prime Agent autonomous runs.
# It proves the local source-to-artifact contract only.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
uv_bin="${UV_BIN:-uv}"

build_dir="$(mktemp -d /tmp/source2agent-gate.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT

"$python_bin" -m pytest
"$python_bin" -m ruff check .
"$uv_bin" lock --check
"$uv_bin" lock --check --project environments/railroad_1959_v1

"$python_bin" -m volume2gym compile \
  --volume-id lantern-ledger-demo \
  --units examples/lantern_ledger/units.json \
  --output "$build_dir/build" \
  --seed 7

"$python_bin" -m volume2gym validate "$build_dir/build"
"$python_bin" -m volume2gym inspect-artifacts "$build_dir/build"
"$python_bin" -m volume2gym reference-eval "$build_dir/build" \
  --output "$build_dir/reference-eval" \
  --split test

test -s "$build_dir/build/manifest.json"
test -s "$build_dir/reference-eval/evaluation.json"

prime_pythonpath="src:environments/railroad_1959_v1"
PYTHONPATH="$prime_pythonpath" "$python_bin" scripts/build_prime_railroad_v1.py --check
PYTHONPATH="$prime_pythonpath" "$python_bin" scripts/evaluate_prime_railroad_v1.py --check
"$python_bin" -m build --wheel --no-isolation \
  --outdir "$build_dir/prime-wheel" \
  environments/railroad_1959_v1 \
  >"$build_dir/prime-wheel.log"
"$python_bin" - "$build_dir/prime-wheel" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel = next(Path(sys.argv[1]).glob("*.whl"))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
required = {
    "railroad_1959_v1/taskset.py",
    "railroad_1959_v1/data/taskset.jsonl.gz",
    "railroad_1959_v1/data/taskset-manifest.json",
}
missing = required - names
if missing:
    raise SystemExit(f"Prime wheel is missing packaged artifacts: {sorted(missing)}")
PY
PYTHONPATH="$prime_pythonpath" "$python_bin" -c \
  'from verifiers.v1.cli.eval.main import main; main()' \
  @ configs/prime/railroad-1959-v1.eval.toml \
  --dry-run True \
  --output-dir "$build_dir/prime-eval-dry-run" \
  >"$build_dir/prime-eval-dry-run.log"
PYTHONPATH="$prime_pythonpath" "$python_bin" -m verifiers.v1.cli.validate \
  @ configs/prime/railroad-1959-v1.validate.toml \
  --only-gold True \
  --output-dir "$build_dir/prime-validate" \
  >"$build_dir/prime-validate.log"
"$python_bin" - "$build_dir/prime-validate/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary["outcomes"] != {
    "error": 0,
    "invalid": 0,
    "missing": 0,
    "timeout": 0,
    "valid": 2742,
}:
    raise SystemExit(f"Prime gold validation failed: {summary}")
PY
cmp "$build_dir/prime-validate/summary.json" \
  evidence/prime/railroad-1959-v1/gold-validation-summary.json

echo "Source2Agent local quality gate: PASS"
