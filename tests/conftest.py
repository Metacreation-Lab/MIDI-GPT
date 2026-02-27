"""Shared pytest fixtures for MIDI-GPT build and API tests.

The ``build_dir`` and ``built_module`` fixtures perform a real CMake configure
+ build cycle using MIDIGPT_NO_TORCH=ON so they can run on a login node without
PyTorch or a GPU.

Prerequisites before running:
    module load protobuf    # or ensure protoc / libprotobuf are on PATH
    pytest tests/
"""

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.resolve()


def _cmake_binary() -> str:
    return os.environ.get("CMAKE", "cmake")


def _run(args: list, cwd=None, extra_env=None, check=True):
    """Run a subprocess, streaming output to the terminal."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check and result.returncode != 0:
        pytest.fail(
            f"Command failed (exit {result.returncode}):\n"
            f"  {' '.join(str(a) for a in args)}\n\n"
            f"{result.stdout}"
        )
    return result


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory):
    """Configure and build the project with MIDIGPT_NO_TORCH=ON.

    Returns the build directory Path.  Skips if essential build tools are
    missing (cmake, a C++ compiler, protoc).
    """
    # Require cmake
    r = subprocess.run(
        [_cmake_binary(), "--version"], capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip("cmake not found — load it with 'module load cmake' first")

    # Require protoc (needed by the protobuf CMakeLists)
    r = subprocess.run(["protoc", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(
            "protoc not found — load it with 'module load protobuf' first"
        )

    bdir = tmp_path_factory.mktemp("build")

    # Configure — pin both PYTHON_EXECUTABLE (pybind11 submodule / old API) and
    # Python3_EXECUTABLE (modern CMake FindPython3) so the built .so matches this
    # interpreter and can be imported by this test session.
    _run(
        [
            _cmake_binary(),
            "-S", str(ROOT),
            "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Debug",   # faster than Release for tests
            "-DMIDIGPT_NO_TORCH=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
    )

    # Build (limit parallelism to avoid swamping login node)
    jobs = os.environ.get("CMAKE_BUILD_JOBS", "4")
    _run(
        [_cmake_binary(), "--build", str(bdir), "-j", jobs]
    )

    return bdir


@pytest.fixture(scope="session")
def built_module(build_dir):
    """Import the midigpt extension built in ``build_dir``.

    The .so sits directly in the build directory.  We insert it into sys.path
    so a plain ``import midigpt`` picks it up.
    """
    build_str = str(build_dir)
    if build_str not in sys.path:
        sys.path.insert(0, build_str)

    # Reload in case a stale version was imported earlier
    if "midigpt" in sys.modules:
        del sys.modules["midigpt"]

    try:
        import midigpt as _mod
    except ImportError as exc:
        pytest.fail(
            f"Could not import midigpt from {build_dir}.\n"
            f"  .so files present: {list(build_dir.glob('midigpt*.so'))}\n"
            f"  Error: {exc}"
        )

    return _mod
