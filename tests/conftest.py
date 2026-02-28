"""Shared pytest fixtures for MIDI-GPT build and API tests.

The ``build_dir`` and ``built_module`` fixtures perform a real CMake configure
+ build cycle (with LibTorch) so the full inference API is available.

Prerequisites before running:
    module load StdEnv/2023 python/3.11.5 gcc/12.3 cmake protobuf/24.4 abseil cuda/12.6
    source /scratch/triana24/.venvs/midigpt/bin/activate
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
    """Configure and build the project with LibTorch enabled.

    Returns the build directory Path.  Skips if essential build tools or
    PyTorch are missing.
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

    # Require torch (needed for LibTorch detection in cmake)
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pytest.skip(
            "torch not importable — load cuda module and activate venv first"
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

    We use ``importlib`` directly to bypass editable-install hooks (e.g.
    scikit-build-core's ``_midigpt_editable.pth``) that would intercept the
    import and load the installed version instead of the fresh build.

    Since the build links LibTorch, torch must be imported before the
    extension .so is dlopen-ed (torch's __init__.py adds its lib/ dir to
    the dynamic linker search path).
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

    # Import torch first so libtorch.so is mapped into the process before
    # _midigpt.so is dlopen-ed.
    import torch  # noqa: F401

    # Pre-register the extension .so from the build dir under both its
    # qualified name (midigpt._midigpt) and bare name (_midigpt) so that
    # the editable-install meta-path hook cannot intercept the relative
    # import inside __init__.py and substitute the installed version.
    ext_spec = importlib.util.spec_from_file_location(
        "midigpt._midigpt",
        str(so_files[0]),
    )
    ext_mod = importlib.util.module_from_spec(ext_spec)
    sys.modules["midigpt._midigpt"] = ext_mod
    sys.modules["_midigpt"] = ext_mod
    try:
        ext_spec.loader.exec_module(ext_mod)
    except (ImportError, OSError) as exc:
        sys.modules.pop("midigpt._midigpt", None)
        sys.modules.pop("_midigpt", None)
        pytest.fail(
            f"Could not load _midigpt extension from {so_files[0]}.\n"
            f"  Error: {exc}"
        )

    # Execute __init__.py with __path__ pinned to pkg_dir.
    # from ._midigpt import * finds the pre-loaded module above.
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
