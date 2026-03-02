"""MIDI pipeline speed benchmarks.

Measures how long midi_to_json_bytes() and midi_to_tokens() take.
This is NOT a pass/fail test — it prints timing results for comparison.

Run:
    python3 -m pytest tests/test_benchmark.py -v -s
"""

import time
import statistics
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    pytest.skip("midigpt not installed", allow_module_level=True)

MIDI_DIR = Path(__file__).parent.parent / "python_scripts_for_testing"
MIDI_TEST_DIR = Path(__file__).parent / "midi_files"
MIDI_FILES = sorted(
    list(MIDI_DIR.glob("*.mid"))
    + list(MIDI_TEST_DIR.glob("**/*.mid"))
)
# Remove empty.mid (has no notes)
MIDI_FILES = [f for f in MIDI_FILES if f.stem != "empty"]
N_ITERATIONS = 100


@pytest.mark.benchmark
class TestParseBenchmark:
    @pytest.mark.parametrize("midi_file", MIDI_FILES,
                             ids=[f.stem for f in MIDI_FILES])
    def test_parse_speed(self, midi_file):
        tc = midigpt.TrainConfig()
        path = str(midi_file)

        # Warm up
        for _ in range(5):
            midigpt.midi_to_json_bytes(path, tc, "{}")

        # Timed runs
        times = []
        for _ in range(N_ITERATIONS):
            start = time.perf_counter()
            midigpt.midi_to_json_bytes(path, tc, "{}")
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        median = statistics.median(times)
        fastest = min(times)

        print(f"\n  [parse] {midi_file.name}: "
              f"mean={mean:.3f}ms  median={median:.3f}ms  "
              f"min={fastest:.3f}ms  stdev={stdev:.3f}ms  "
              f"(n={N_ITERATIONS})")


# Subset of files that can be tokenized (skip very long files)
TOKENIZABLE_FILES = [f for f in MIDI_FILES if f.stem not in {
    "Maestro_1", "Maestro_2", "Maestro_3", "Maestro_4", "Maestro_5",
    "Maestro_6", "Maestro_7", "Maestro_8", "Maestro_9", "Maestro_10",
}]

FROZEN_CONFIG = {
    "both_in_one": True,
    "unquantized": False,
    "do_multi_fill": False,
    "use_velocity_levels": True,
    "use_microtiming": True,
    "transpose": 0,
    "resolution": 12,
    "decode_resolution": 1920,
    "decode_final": False,
    "delta_resolution": 1920,
}


@pytest.mark.benchmark
class TestTokenizeBenchmark:
    @pytest.mark.parametrize("midi_file", TOKENIZABLE_FILES,
                             ids=[f.stem for f in TOKENIZABLE_FILES])
    def test_tokenize_speed(self, midi_file):
        enc = midigpt.ExpressiveEncoder()
        for key, value in FROZEN_CONFIG.items():
            setattr(enc.config, key, value)
        path = str(midi_file)

        # Verify it works before benchmarking
        try:
            enc.midi_to_tokens(path)
        except RuntimeError:
            pytest.skip(f"{midi_file.name} cannot be tokenized")

        # Warm up
        for _ in range(5):
            enc.midi_to_tokens(path)

        # Timed runs
        times = []
        for _ in range(N_ITERATIONS):
            start = time.perf_counter()
            enc.midi_to_tokens(path)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        median = statistics.median(times)
        fastest = min(times)

        print(f"\n  [tokenize] {midi_file.name}: "
              f"mean={mean:.3f}ms  median={median:.3f}ms  "
              f"min={fastest:.3f}ms  stdev={stdev:.3f}ms  "
              f"(n={N_ITERATIONS})")
