# MIDI-GPT Build & Test Makefile
#
# Prerequisites
#   macOS : brew install cmake abseil protobuf
#   Debian: apt install cmake libprotobuf-dev protobuf-compiler libabsl-dev
#   HPC   : module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4
#           source /scratch/triana24/.venvs/midigpt/bin/activate

.PHONY: install install-osc install-train install-symusic install-notorch \
        golden test test-realtime benchmark verify clean

# Full install with inference + training dependencies
install:
	pip install -e ".[train]"

# Install with OSC real-time server support
install-osc:
	pip install -e ".[train,osc]"

# Training only (no inference/torch)
install-notorch:
	CMAKE_ARGS="-DMIDIGPT_NO_TORCH=ON" pip install -e ".[train]"

# Symusic MIDI backend
install-symusic:
	CMAKE_ARGS="-DMIDIGPT_MIDI_BACKEND=symusic" pip install -e ".[train]"

# Generate all golden reference files (run before making encoder changes)
golden:
	MIDIGPT_GENERATE_GOLDEN=1 python3 -m pytest \
		tests/test_golden.py \
		tests/test_backend_parity.py \
		tests/test_tokenization_parity.py \
		tests/test_valid_segments.py \
		-v

# Run all tests (excluding benchmarks and inference tests)
test:
	python3 -m pytest tests/ -v -m "not benchmark and not inference"

# Run real-time generation tests only
test-realtime:
	python3 -m pytest tests/test_realtime.py -v

# Run benchmarks
benchmark:
	python3 -m pytest tests/test_benchmark.py -v -s

# Full verification
verify: test benchmark

# Clean build artifacts
clean:
	rm -rf build/ _skbuild/ *.egg-info/ .pytest_cache/
