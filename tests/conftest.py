"""Shared pytest fixtures for MIDI-GPT build and API tests.

The ``build_dir`` and ``built_module`` fixtures perform a real CMake configure
+ build cycle using MIDIGPT_NO_TORCH=ON so they can run on a login node without
PyTorch or a GPU.

Prerequisites before running:
    module load protobuf    # or ensure protoc / libprotobuf are on PATH
    pytest tests/
"""

import importlib.util
import os
import subprocess
import sys
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
    """Import the midigpt package assembled in ``build_dir/midigpt/``.

    The CMake post-build step copies ``__init__.py`` and ``_midigpt.so``
    into ``build_dir/midigpt/`` so pytest can exercise the freshly compiled
    extension without a full ``pip install``.

    We use ``importlib`` directly instead of a bare ``import`` so that any
    editable-install hooks registered by a previously installed midigpt wheel
    (e.g. scikit-build-core's ``_midigpt_editable.pth``) cannot intercept the
    import and accidentally load the wrong shared library.
    """
    pkg_dir = build_dir / "midigpt"
    pkg_init = pkg_dir / "__init__.py"

    # Clear stale cached modules from any previous test session or editable install.
    for mod_name in list(sys.modules):
        if mod_name == "midigpt" or mod_name.startswith("midigpt."):
            del sys.modules[mod_name]
    sys.modules.pop("_midigpt", None)

    so_files = list(pkg_dir.glob("_midigpt*.so")) if pkg_dir.exists() else []

    if not pkg_init.exists():
        pytest.fail(
            f"__init__.py not found at {pkg_init}.\n"
            "Check the CMake POST_BUILD copy command."
        )
    if not so_files:
        pytest.fail(
            f"No _midigpt*.so found in {pkg_dir}.\n"
            f"build_dir contents: {list(build_dir.iterdir())}"
        )

    # Load via spec so the package's __path__ is correctly set to pkg_dir,
    # allowing the relative import `from ._midigpt import *` inside __init__.py
    # to find the freshly built extension .so.
    spec = importlib.util.spec_from_file_location(
        "midigpt",
        str(pkg_init),
        submodule_search_locations=[str(pkg_dir)],
    )
    _mod = importlib.util.module_from_spec(spec)
    sys.modules["midigpt"] = _mod

    try:
        spec.loader.exec_module(_mod)
    except (ImportError, OSError) as exc:
        del sys.modules["midigpt"]
        pytest.fail(
            f"Could not import midigpt from {build_dir}.\n"
            f"  _midigpt*.so files: {so_files}\n"
            f"  Error: {exc}"
        )

    return _mod
