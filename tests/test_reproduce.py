
"""Tests for the isolated model reproduction workflow."""

import subprocess
import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest
import yaml

from maintain import reproduce


@pytest.fixture
def project_folder(tmp_path, monkeypatch):
    """Create a small project layout for safe reproduction tests."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    config = {
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": version("scikit-learn"),
        }
    }

    (config_dir / "decision.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    monkeypatch.setattr(reproduce, "PROJECT_ROOT", tmp_path)

    return tmp_path


def test_matching_environment_is_accepted(project_folder):
    """The recorded Python and scikit-learn versions should pass."""

    reproduce.check_environment()


def test_different_sklearn_version_is_rejected(
    project_folder, monkeypatch
):
    """Reproduction must reject an incompatible ML environment."""

    monkeypatch.setattr(
        reproduce,
        "version",
        lambda package: "0.0.0",
    )

    with pytest.raises(RuntimeError, match="Environment differs"):
        reproduce.check_environment()


def test_tracked_files_are_copied(
    project_folder, tmp_path, monkeypatch
):
    """Copy the expected source files into a separate workspace."""

    source = project_folder / "src" / "maintain" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 42\n", encoding="utf-8")

    def fake_git(command, **kwargs):
        return SimpleNamespace(
            stdout=b"src/maintain/example.py\0"
        )

    monkeypatch.setattr(reproduce.subprocess, "run", fake_git)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    reproduce.copy_tracked_files(workspace)

    destination = workspace / "src" / "maintain" / "example.py"

    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == "VALUE = 42\n"
    assert source.is_file()


def test_unsafe_tracked_path_is_rejected(
    project_folder, tmp_path, monkeypatch
):
    """A Git path must not escape the temporary workspace."""

    def fake_git(command, **kwargs):
        return SimpleNamespace(stdout=b"../outside.txt\0")

    monkeypatch.setattr(reproduce.subprocess, "run", fake_git)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="Unsafe repository path"):
        reproduce.copy_tracked_files(workspace)


def test_missing_tracked_file_is_rejected(
    project_folder, tmp_path, monkeypatch
):
    """An incomplete repository snapshot must not be accepted."""

    def fake_git(command, **kwargs):
        return SimpleNamespace(stdout=b"missing_file.py\0")

    monkeypatch.setattr(reproduce.subprocess, "run", fake_git)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError, match="Tracked file missing"):
        reproduce.copy_tracked_files(workspace)


def test_reproduction_steps_use_the_correct_workspace(
    project_folder, tmp_path, monkeypatch
):
    """Subcommands must run in the selected isolated directory."""

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["options"] = kwargs

    monkeypatch.setattr(reproduce.subprocess, "run", fake_run)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    environment = {
        "PYTHONPATH": str(workspace / "src"),
    }

    reproduce.run_step(
        workspace,
        environment,
        "Test step",
        ["maintain.splits"],
    )

    assert captured["command"] == [
        sys.executable,
        "-m",
        "maintain.splits",
    ]
    assert captured["options"]["cwd"] == workspace
    assert captured["options"]["env"] == environment
    assert captured["options"]["check"] is True


def test_reproduction_rejects_imports_from_original_project(
    project_folder, monkeypatch
):
    """Stop if Python loads code outside the temporary workspace."""

    monkeypatch.setattr(
        reproduce,
        "check_environment",
        lambda: None,
    )

    monkeypatch.setattr(
        reproduce,
        "copy_tracked_files",
        lambda workspace: None,
    )

    def fake_import_check(command, **kwargs):
        return SimpleNamespace(stdout=str(project_folder))

    monkeypatch.setattr(
        reproduce.subprocess,
        "run",
        fake_import_check,
    )

    with pytest.raises(
        RuntimeError,
        match="not importing the temporary project copy",
    ):
        reproduce.reproduce()
