from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import volume2gym
from volume2gym.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return environment


def test_version_license_and_console_entry_point_metadata_are_consistent() -> None:
    project = _project_metadata()

    assert project["name"] == "volume2gym"
    assert project["version"] == volume2gym.__version__
    assert project["requires-python"] == ">=3.11"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["scripts"]["v2g"] == "volume2gym.cli:main"
    assert callable(main)


def test_python_module_help_is_available_without_optional_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "volume2gym", "--help"],
        cwd=ROOT,
        env=_source_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "compile" in completed.stdout
    assert "validate" in completed.stdout
    assert "inspect-artifacts" in completed.stdout
    assert "export" in completed.stdout
    assert "reference-eval" in completed.stdout


def test_python_module_propagates_cli_failure_status(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "volume2gym", "validate", str(tmp_path / "missing")],
        cwd=ROOT,
        env=_source_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "v2g: error:" in completed.stderr
    assert "manifest.json" in completed.stderr
