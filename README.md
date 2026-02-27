[![N|Solid](https://drive.google.com/uc?export=view&id=1u4xiWN3s0PAii8zn3-qxJ7wn35tBOypY)](https://metacreation.net/category/projects/)

# MIDI-GPT

Generative system based on the Transformer architecture for computer-assisted music composition.
Paper: [AAAI 2025](https://arxiv.org/abs/2501.17011)

---

## Dependencies

| Dependency | Source |
|---|---|
| CMake ≥ 3.21 | system / module |
| C++20 compiler (GCC ≥ 11 or Clang ≥ 14) | system / module |
| Protobuf (libprotobuf + protoc) | system / module |
| Python ≥ 3.10 | system / module |
| PyTorch ≥ 2.5.1 | pip |
| pybind11 ≥ 2.12 | pip (or git submodule) |
| midifile | git submodule |

---

## Build

### Prerequisites

```bash
git submodule update --init --recursive
```

### Python install (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .              # development / editable install
# or
pip install -e ".[train]"     # + training dependencies
```

Verify:

```bash
python -c "import midigpt; print(midigpt.version())"
```

### Wheel build

```bash
pip install build
python -m build --wheel
```

### C++-only (no Python required)

Requires Protobuf headers and `protoc` on `PATH`.

```bash
cmake -S . -B build -DMIDIGPT_NO_TORCH=ON
cmake --build build -j$(nproc)
```

With Torch (manual LibTorch path):

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j$(nproc)
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `MIDIGPT_NO_TORCH` | `OFF` | Exclude LibTorch — dataset creation and training only |
| `MIDIGPT_TRACE` | `OFF` | Enable `-finstrument-functions` tracing |

---

## HPC (Fir / Alliance Canada)

```bash
module load StdEnv/2023 python/3.11 gcc/12 cmake protobuf

python -m venv $SCRATCH/.venvs/midigpt
source $SCRATCH/.venvs/midigpt/bin/activate
pip install torch
pip install -e ".[train]"
```

C++-only (training only, no inference):

```bash
cmake -S . -B $SCRATCH/midigpt-build -DMIDIGPT_NO_TORCH=ON
cmake --build $SCRATCH/midigpt-build -j$(nproc)
```

---

## Usage

### Inference

See `python_scripts_for_testing/pythoninferencetest.py`.

Unzip the model first:

```bash
unzip models/model.zip -d models/
```

### Dataset creation

```bash
python python_scripts/create_dataset.py \
    --encoding EXPRESSIVE_ENCODER \
    --data_dir /path/to/midi/{train,test,valid} \
    --output /path/to/output.arr
```

### Training

```bash
python python_scripts/train.py \
    --arch gpt2 \
    --config python_scripts/config/gpt2.json \
    --encoding EXPRESSIVE_ENCODER \
    --dataset /path/to/output.arr \
    --batch_size 32 \
    --label my_run
```

### Checkpoint → TorchScript

```bash
python python_scripts/convert.py --checkpoint /path/to/checkpoint
```

---

## Repository layout

```
CMakeLists.txt               top-level CMake (single authoritative build)
cmake/
  dependencies.cmake         all find_package / add_subdirectory calls
pyproject.toml               pip / wheel build (scikit-build-core backend)
src/
  cpp/
    bindings/
      lib.cpp                pybind11 PYBIND11_MODULE definition
  common/                    shared C++ (encoders, MIDI I/O, data structures)
  dataset_creation/          LZ4 compression, file I/O
  inference/                 sampling loop, Jagged dataset reader
include/                     public C++ headers
libraries/
  protobuf/                  .proto files + generated CMake library
  midifile/                  git submodule — MIDI file parser
  pybind11/                  git submodule — Python bindings
python_scripts/              training, dataset creation, conversion scripts
pip_requirements/            per-task requirement files (HPC manual installs)
```

## CMake targets

| Target | Type | Description |
|---|---|---|
| `midigpt_core` | STATIC library | Pure C++ — encoder, MIDI I/O, sampling, dataset I/O |
| `midigpt` | Shared module | pybind11 Python extension (`import midigpt`) |
| `midigpt_proto` | STATIC library | Protobuf generated C++ (from `libraries/protobuf/`) |
| `midifile` | STATIC library | MIDI file parser (from `libraries/midifile/`) |
| `midigpt_tracer` | STATIC library | Only built when `MIDIGPT_TRACE=ON` |
