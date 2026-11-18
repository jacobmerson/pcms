"""
Run the Python API examples

An example whose optional dependencies are missing
is skipped. Set PCMS_REQUIRE_EXAMPLES=1 to require
those examples to run.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import pcms
import PyOmega_h as omega_h

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "python-api-example"
FIELDS_DIR = REPO_ROOT / "examples" / "fields"
EXAMPLE_TIMEOUT_SECONDS = 600


def _test_data_dir():
    return pathlib.Path(
        os.environ.get("PCMS_TEST_DATA_DIR", REPO_ROOT / "pcms_testcases")
    )


def _skip_or_fail(reason):
    if os.environ.get("PCMS_REQUIRE_EXAMPLES", "0") == "1":
        pytest.fail(f"{reason} (PCMS_REQUIRE_EXAMPLES=1)")
    pytest.skip(reason)


def _run_example(script, *args):
    path = script if isinstance(script, pathlib.Path) else EXAMPLES_DIR / script
    result = subprocess.run(
        [sys.executable, str(path), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=EXAMPLE_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"{script} exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


def test_conservative_transfer_example():
    if not hasattr(pcms, "create_conservative_projection"):
        _skip_or_fail("pcms was built without PETSc/MeshFields")
    stdout = _run_example("example_conservative_transfer.py")
    assert "Field transfer completed" in stdout


def test_flow_example():
    if not hasattr(omega_h, "exodus_open"):
        _skip_or_fail("PyOmega_h was built without Exodus support")
    exo_path = _test_data_dir() / "exodus" / "flow" / "flow.exo"
    if not exo_path.exists():
        _skip_or_fail(f"test data not found: {exo_path}")
    with open(exo_path, "rb") as f:
        if f.read(64).startswith(b"version https://git-lfs"):
            _skip_or_fail(f"{exo_path} is a Git LFS pointer; fetch it with "
                          "git lfs pull")
    stdout = _run_example("flow_example.py", exo_path)
    assert "SUCCESS" in stdout


# The fields track of examples/: every example asserts its own result and
# exits non-zero on a failed check, so running it is the test.
FIELDS_EXAMPLES = sorted(p.parent.name for p in FIELDS_DIR.glob("*/main.py"))


@pytest.mark.parametrize("name", FIELDS_EXAMPLES)
def test_fields_example(name):
    if name != "hello-field" and not hasattr(pcms.backend, "omega_h"):
        _skip_or_fail("pcms was built without Omega_h")
    _run_example(FIELDS_DIR / name / "main.py")
