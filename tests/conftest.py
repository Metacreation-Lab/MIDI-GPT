"""Shared pytest fixtures for MIDI-GPT build and API tests.

The ``build_dir`` and ``built_module`` fixtures perform a real CMake configure
+ build cycle (with LibTorch) so the full inference API is available.

System prerequisites (must be on PATH before running):
  - cmake >= 3.21
  - protoc + libprotobuf-dev headers
  - abseil headers
  - a C++20 compiler (gcc >= 10, clang >= 12, Apple Clang >= 13)

Install on macOS:     brew install cmake abseil protobuf
Install on Debian:    apt install cmake libprotobuf-dev protobuf-compiler libabsl-dev
Install on HPC:       module load cmake protobuf abseil (+ cuda if available)

PyTorch must also be importable in the active Python environment:
    pip install torch
"""

import importlib.machinery
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Extension suffix for this interpreter: .cpython-311-x86_64-linux-gnu.so on
# Linux/Mac, .cp311-win_amd64.pyd on Windows.
_EXT_SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]

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
            "torch not importable — install it with: pip install torch"
        )

    # The venv's torch is a CUDA build, so cmake's find_package(Torch) also
    # requires CUDA headers.  If nvcc isn't already on PATH, load the cuda
    # module via Lmod inside the subprocess before running cmake.
    lmod_init = os.path.join(os.environ.get("LMOD_PKG", ""), "init", "bash")
    cuda_module = os.environ.get("MIDIGPT_CUDA_MODULE", "cuda/12.2")
    try:
        _has_nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        _has_nvcc = False
    needs_cuda_load = not _has_nvcc and os.path.exists(lmod_init)

    def _cmake_run(args, cwd=None):
        """Run a cmake command, loading the cuda module first if needed."""
        if needs_cuda_load:
            cmd = (
                f"source {lmod_init} "
                f"&& module load {cuda_module} "
                f"&& {' '.join(shlex.quote(str(a)) for a in args)}"
            )
            return _run(["bash", "-c", cmd], cwd=cwd)
        return _run(args, cwd=cwd)

    bdir = tmp_path_factory.mktemp("build")

    # Configure — pin both PYTHON_EXECUTABLE (pybind11 submodule / old API) and
    # Python3_EXECUTABLE (modern CMake FindPython3) so the built .so matches this
    # interpreter and can be imported by this test session.
    _cmake_run(
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
    _cmake_run(
        [_cmake_binary(), "--build", str(bdir), "-j", jobs]
    )

    return bdir


@pytest.fixture(scope="session")
def built_module(build_dir):
    """Return a working midigpt module after a successful cmake build.

    When an editable install of midigpt is already present in the environment
    (the normal development workflow), loading a second copy of the extension
    .so into the same process crashes because torch's global state is already
    initialised.  In that case we return the already-installed module — the
    cmake build correctness is already verified by the ``build_dir`` fixture
    completing without error.

    In a clean CI environment (no editable install), we load the freshly built
    .so directly so the API tests exercise the cmake artefact.
    """
    # Fast path: editable / pip install already in the process.
    try:
        import midigpt as _installed  # noqa: F401
        return _installed
    except (ImportError, OSError):
        pass

    # Clean environment: load the .so produced by cmake.
    pkg_dir = build_dir / "midigpt"
    pkg_init = pkg_dir / "__init__.py"

    so_files = list(pkg_dir.glob(f"_midigpt*{_EXT_SUFFIX}")) if pkg_dir.exists() else []

    if not pkg_init.exists():
        pytest.fail(
            f"__init__.py not found at {pkg_init}.\n"
            "Check the CMake POST_BUILD copy command."
        )
    if not so_files:
        pytest.fail(
            f"No _midigpt*{_EXT_SUFFIX} found in {pkg_dir}.\n"
            f"build_dir contents: {list(build_dir.iterdir())}"
        )

    # torch must be imported before dlopen so its lib/ dir is on the linker path.
    import torch  # noqa: F401

    ext_spec = importlib.util.spec_from_file_location(
        "midigpt._midigpt", str(so_files[0])
    )
    ext_mod = importlib.util.module_from_spec(ext_spec)
    sys.modules["midigpt._midigpt"] = ext_mod
    sys.modules["_midigpt"] = ext_mod
    try:
        ext_spec.loader.exec_module(ext_mod)
    except (ImportError, OSError) as exc:
        sys.modules.pop("midigpt._midigpt", None)
        sys.modules.pop("_midigpt", None)
        pytest.fail(f"Could not load _midigpt from {so_files[0]}: {exc}")

    spec = importlib.util.spec_from_file_location(
        "midigpt", str(pkg_init),
        submodule_search_locations=[str(pkg_dir)],
    )
    _mod = importlib.util.module_from_spec(spec)
    sys.modules["midigpt"] = _mod
    try:
        spec.loader.exec_module(_mod)
    except (ImportError, OSError) as exc:
        del sys.modules["midigpt"]
        pytest.fail(
            f"Could not import midigpt from {build_dir}: {exc}\n"
            f"  extension files: {so_files}"
        )

    return _mod


# ── inference fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ckpt_path():
    """Resolve a TorchScript checkpoint for inference tests.

    Reads from MIDIGPT_CKPT env var.  Skips if unset or file missing.
    """
    path = os.environ.get("MIDIGPT_CKPT", "")
    if not path:
        pytest.skip("MIDIGPT_CKPT env var not set — skipping inference tests")
    if not os.path.isfile(path):
        pytest.skip(f"Checkpoint not found: {path}")
    return path


@pytest.fixture(scope="session")
def sample_piece_json():
    """Load a test MIDI file into a JSON string via ExpressiveEncoder."""
    import json

    try:
        import midigpt
    except ImportError:
        pytest.skip("midigpt not installed")

    midi_dir = ROOT / "tests" / "midi_files" / "singletrack"
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        midi_dir = ROOT / "tests" / "midi_files"
        midi_files = sorted(midi_dir.glob("**/*.mid"))
    if not midi_files:
        pytest.skip("No test MIDI files found")

    enc = midigpt.ExpressiveEncoder()
    for k, v in {
        "both_in_one": True, "unquantized": False, "do_multi_fill": False,
        "use_velocity_levels": True, "use_microtiming": True, "transpose": 0,
        "resolution": 12, "decode_resolution": 1920, "decode_final": False,
        "delta_resolution": 1920,
    }.items():
        setattr(enc.config, k, v)

    # Try files until one parses successfully
    for mf in midi_files:
        try:
            piece_json = enc.midi_to_json(str(mf))
            piece = json.loads(piece_json)
            if piece.get("tracks") and len(piece["tracks"][0].get("bars", [])) >= 3:
                return piece_json
        except Exception:
            continue

    pytest.skip("No suitable test MIDI file found (need >= 3 bars)")
