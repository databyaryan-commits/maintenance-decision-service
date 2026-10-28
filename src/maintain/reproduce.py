
"""Reproduce the frozen maintenance model in a temporary workspace."""

import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from maintain.data import PROJECT_ROOT


def check_environment():
    """Require the Python and scikit-learn versions used for evaluation."""

    config_path = PROJECT_ROOT / "configs" / "decision.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    expected = config["environment"]
    python_version = sys.version.split()[0]
    sklearn_version = version("scikit-learn")

    print("ENVIRONMENT CHECK", flush=True)
    print("-" * 55, flush=True)
    print(f"Python: {python_version}", flush=True)
    print(f"Scikit-learn: {sklearn_version}", flush=True)

    if (
        python_version != expected["python"]
        or sklearn_version != expected["scikit_learn"]
    ):
        raise RuntimeError(
            "Environment differs from the frozen experiment. "
            "Use the documented Python and dependency versions."
        )


def copy_tracked_files(workspace):
    """Copy the committed repository files into an empty workspace."""

    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )

    filenames = [
        os.fsdecode(name)
        for name in result.stdout.split(b"\0")
        if name
    ]

    if not filenames:
        raise RuntimeError("No tracked repository files were found.")

    for filename in filenames:
        relative_path = Path(filename)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Unsafe repository path: {filename}")

        source = PROJECT_ROOT / relative_path
        destination = workspace / relative_path

        if not source.is_file():
            raise FileNotFoundError(
                f"Tracked file missing from working tree: {source}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # Models and raw data are deliberately absent from the snapshot.
    for relative_path in (
        "models/random_forest.joblib",
        "data/raw/ai4i2020.csv",
    ):
        if (workspace / relative_path).exists():
            raise RuntimeError(
                f"Unexpected generated file in snapshot: {relative_path}"
            )

    print(f"Copied {len(filenames)} tracked files.", flush=True)


def run_step(workspace, environment, title, arguments):
    """Run one existing project command in the isolated workspace."""

    print(f"\n=== {title} ===", flush=True)

    subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=workspace,
        env=environment,
        check=True,
    )


def reproduce():
    """Build and verify the frozen model without touching original artifacts."""

    check_environment()

    print("\nAUTOMATED REPRODUCTION", flush=True)
    print("-" * 55, flush=True)
    print("Creating a temporary workspace...", flush=True)

    with TemporaryDirectory(prefix="maintenance-reproduce-") as folder:
        workspace = Path(folder).resolve()
        copy_tracked_files(workspace)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(workspace / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        # Check imports before any command creates or overwrites files.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from maintain.data import PROJECT_ROOT; print(PROJECT_ROOT)",
            ],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )

        imported_root = Path(result.stdout.strip()).resolve()

        if imported_root != workspace:
            raise RuntimeError(
                "Python is not importing the temporary project copy."
            )

        print(f"Isolated project: {workspace}", flush=True)

        run_step(
            workspace,
            environment,
            "1. Download and verify dataset",
            ["maintain.download_data"],
        )

        run_step(
            workspace,
            environment,
            "2. Recreate data splits",
            ["maintain.splits"],
        )

        run_step(
            workspace,
            environment,
            "3. Reproduce trained models",
            ["maintain.train"],
        )

        run_step(
            workspace,
            environment,
            "4. Verify frozen model and checksum",
            ["maintain.frozen"],
        )

        run_step(
            workspace,
            environment,
            "5. Verify API prediction parity",
            ["maintain.check_parity"],
        )

        run_step(
            workspace,
            environment,
            "6. Run complete test suite",
            ["pytest", "-q"],
        )

        print("\nREPRODUCTION SUCCESSFUL", flush=True)
        print("-" * 55, flush=True)
        print("Dataset verification: PASS", flush=True)
        print("Model checksum verification: PASS", flush=True)
        print("API prediction parity: PASS", flush=True)
        print("Automated tests: PASS", flush=True)

    print("\nTemporary workspace cleaned up.", flush=True)
    print("Original project artifacts were not modified.", flush=True)


if __name__ == "__main__":
    reproduce()
