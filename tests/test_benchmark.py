"""MIDI parsing speed benchmark.

Measures how long midi_to_json_bytes() takes to parse a MIDI file.
This is NOT a pass/fail test — it prints timing results for comparison
between the midifile and symusic backends.

Run:
    # With whichever backend is currently installed:
    python3 -m pytest tests/test_benchmark.py -v -s

    # Or compare both backends:
    # 1) Install with midifile (default), run benchmark
    # 2) Install with symusic, run benchmark
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

        print(f"\n  {midi_file.name}: "
              f"mean={mean:.3f}ms  median={median:.3f}ms  "
              f"min={fastest:.3f}ms  stdev={stdev:.3f}ms  "
              f"(n={N_ITERATIONS})")
