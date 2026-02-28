"""Compilation tests.

These tests verify that the CMake build system produces the expected artifacts
and responds correctly to options.  They perform real cmake configure + build
steps using a temporary build directory.

Run:
    pytest tests/test_compilation.py -v
    (ensure cmake, protoc/protobuf headers, and torch are available first)
"""

import importlib.machinery
import os
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, _EXT_SUFFIX, _cmake_binary, _run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure(bdir: Path, *extra_args, expect_failure=False):
    """Run cmake configure step, return the CompletedProcess."""
    jobs_arg = os.environ.get("CMAKE_BUILD_JOBS", "4")
    result = _run(
        [
            _cmake_binary(),
            "-S", str(ROOT),
            "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Debug",
            *extra_args,
        ],
        check=not expect_failure,
    )
    return result


def _build(bdir: Path, expect_failure=False):
    jobs = os.environ.get("CMAKE_BUILD_JOBS", "4")
    result = _run(
        [_cmake_binary(), "--build", str(bdir), "-j", jobs],
        check=not expect_failure,
    )
    return result


# ---------------------------------------------------------------------------
# cmake --version guard (skip entire module if cmake/protoc missing)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    pass  # guards are inside each test via skip marks


# ---------------------------------------------------------------------------
# Configure-only tests (fast)
# ---------------------------------------------------------------------------


class TestCMakeConfigure:
    """Verify the cmake configure step succeeds for all option combinations."""

    def test_configure_no_torch(self, tmp_path):
        """Default target: no Torch, should always succeed when protoc is present."""
        r = subprocess.run(["protoc", "--version"], capture_output=True)
        if r.returncode != 0:
            pytest.skip("protoc not found — module load protobuf")

        _configure(tmp_path, "-DMIDIGPT_NO_TORCH=ON")
        # CMakeCache.txt is created on success
        assert (tmp_path / "CMakeCache.txt").exists()

    def test_configure_trace_flag(self, tmp_path):
        """MIDIGPT_TRACE=ON should configure without error (no actual build)."""
        r = subprocess.run(["protoc", "--version"], capture_output=True)
        if r.returncode != 0:
            pytest.skip("protoc not found — module load protobuf")

        _configure(tmp_path, "-DMIDIGPT_NO_TORCH=ON", "-DMIDIGPT_TRACE=ON")
        assert (tmp_path / "CMakeCache.txt").exists()

    def test_cmake_version_requirement(self, tmp_path):
        """Project requires CMake ≥ 3.21; verify the installed cmake satisfies it."""
        result = subprocess.run(
            [_cmake_binary(), "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0, "cmake not found"
        # first line: "cmake version X.Y.Z"
        version_str = result.stdout.splitlines()[0].split()[-1]
        major, minor, *_ = (int(x) for x in version_str.split("."))
        assert (major, minor) >= (3, 21), (
            f"CMake {version_str} < 3.21 required by project"
        )

    def test_submodules_present(self):
        """Fail loudly if git submodules are not checked out."""
        pybind = ROOT / "libraries" / "pybind11" / "CMakeLists.txt"
        midifile = ROOT / "libraries" / "midifile" / "CMakeLists.txt"
        assert pybind.exists(), (
            f"pybind11 submodule missing at {pybind}.\n"
            "Run: git submodule update --init --recursive"
        )
        assert midifile.exists(), (
            f"midifile submodule missing at {midifile}.\n"
            "Run: git submodule update --init --recursive"
        )

    def test_cmake_cache_contains_targets(self, tmp_path):
        """After configure, the cache should mention our custom options."""
        r = subprocess.run(["protoc", "--version"], capture_output=True)
        if r.returncode != 0:
            pytest.skip("protoc not found — module load protobuf")

        _configure(tmp_path, "-DMIDIGPT_NO_TORCH=ON")
        cache = (tmp_path / "CMakeCache.txt").read_text()
        assert "MIDIGPT_NO_TORCH" in cache
        assert "MIDIGPT_TRACE" in cache


# ---------------------------------------------------------------------------
# Build tests (slower — compile everything)
# ---------------------------------------------------------------------------


class TestBuild:
    """Full cmake configure + build cycle."""

    def test_build_succeeds(self, build_dir):
        """The session-scoped ``build_dir`` fixture already ran this build.
        If the fixture succeeded, this test trivially passes.  It exists to
        produce a named test entry in the report.
        """
        assert build_dir.exists()

    def test_extension_so_exists(self, build_dir):
        """The pybind11 extension must be present inside build_dir/midigpt/ after build."""
        pkg_dir = build_dir / "midigpt"
        so_files = list(pkg_dir.glob(f"_midigpt*{_EXT_SUFFIX}")) if pkg_dir.exists() else []
        assert so_files, (
            f"No _midigpt*{_EXT_SUFFIX} found in {pkg_dir}.\n"
            f"build_dir contents: {list(build_dir.iterdir())}"
        )

    def test_extension_so_is_not_empty(self, build_dir):
        pkg_dir = build_dir / "midigpt"
        so_files = list(pkg_dir.glob(f"_midigpt*{_EXT_SUFFIX}")) if pkg_dir.exists() else []
        assert so_files, f"No _midigpt*{_EXT_SUFFIX} (see test_extension_so_exists)"
        size = so_files[0].stat().st_size
        assert size > 1024, f"Extension is suspiciously small: {size} bytes"

    def test_package_init_exists(self, build_dir):
        """The post-build step must copy __init__.py into build_dir/midigpt/."""
        init_file = build_dir / "midigpt" / "__init__.py"
        assert init_file.exists(), (
            f"__init__.py not found at {init_file}.\n"
            "Check the CMake POST_BUILD copy command in CMakeLists.txt."
        )

    def test_core_static_lib_exists(self, build_dir):
        """midigpt_core must be built as a static library."""
        libs = list(build_dir.glob("**/libmidigpt_core.a"))
        assert libs, (
            f"libmidigpt_core.a not found under {build_dir}.\n"
            "Check that the midigpt_core STATIC target is defined in CMakeLists.txt"
        )

    def test_proto_static_lib_exists(self, build_dir):
        """midigpt_proto must be built (protobuf .proto → .pb.cc → .a)."""
        libs = list(build_dir.glob("**/libmidigpt_proto.a"))
        assert libs, (
            f"libmidigpt_proto.a not found under {build_dir}.\n"
            "Check libraries/protobuf/CMakeLists.txt"
        )

    def test_generated_protobuf_headers(self, build_dir):
        """CMake must have generated .pb.h files from the .proto sources."""
        pb_headers = list(build_dir.rglob("*.pb.h"))
        assert pb_headers, (
            f"No generated *.pb.h files found under {build_dir}.\n"
            "protobuf_generate_cpp() may have failed"
        )
        names = {f.name for f in pb_headers}
        # At minimum the main MIDI proto must be present
        assert "midi.pb.h" in names, (
            f"midi.pb.h missing from generated headers.\nFound: {names}"
        )

    def test_module_importable(self, built_module):
        """The compiled extension must be importable without error."""
        assert built_module is not None

    def test_torch_api_present(self, built_module):
        """With LibTorch enabled, the inference API must be exposed."""
        assert hasattr(built_module, "sample_multi_step"), (
            "sample_multi_step is missing — LibTorch may not have been linked.\n"
            "Ensure the cuda module is loaded before building."
        )
