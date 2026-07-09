# Training

This page covers training a new midigpt model from scratch, or fine-tuning an existing checkpoint on your own dataset.

## Requirements

```bash
pip install "midigpt[train]"
```

Training requires PyTorch Lightning, HuggingFace `datasets`, and `pyarrow`. The C++ MIDI parser is not fork-safe, so `num_workers` must be `0`.

---

## Data format

midigpt trains on MIDI files stored as **parquet shards**. Each row in the parquet file represents one MIDI piece. The recommended source is [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI), available on HuggingFace Datasets.

---

## Step 1 — Preprocess parquet shards

The preprocessing step builds a valid-index cache so dataset initialization is instant on every subsequent run. It runs a fast metadata filter (pure PyArrow, no MIDI parsing) followed by per-row validation via an isolated subprocess that bisects on crashes.

```bash
python -m midigpt.training.preprocess \
    --parquet /data/train/*.parquet \
    --checkpoint models/yellow_medium-final.safetensors
```

Or supply a raw encoder config JSON instead of a checkpoint:

```bash
python -m midigpt.training.preprocess \
    --parquet /data/train/*.parquet \
    --encoder-config models/yellow_encoder.json \
    --min-bars 4 --min-tracks 1
```

Index files are cached in `~/.midigpt/` (override with the `MIDIGPT_CACHE` environment variable).

---

## Step 2 — Launch training

**Command line:**

```bash
python -m midigpt.training.trainer \
    --config      models/train_config.json \
    --train-data  /data/train/*.parquet \
    --eval-data   /data/valid/*.parquet \
    --output-dir  checkpoints/run_001
```

**Python API:**

```python
from midigpt.training.trainer import TrainConfig, train

config = TrainConfig.from_file("models/train_config.json")
config.output_dir = "checkpoints/run_001"

train(
    config,
    train_path="/data/train/00000.parquet",
    eval_path="/data/valid/00000.parquet",
)
```

`train()` uses PyTorch Lightning internally. At the end of training it writes a packed `.safetensors` bundle (`model_final.safetensors`) containing the weights, architecture config, and encoder config. Intermediate checkpoints are saved every `save_steps` steps.

---

## `TrainConfig` reference

### Architecture

| Field | Default | Description |
|---|---|---|
| `encoder_config_path` | `""` | Path to an encoder `.json` or a packed `.pt` bundle |
| `n_embd` | `512` | Embedding dimension |
| `n_layer` | `6` | Number of transformer layers |
| `n_head` | `8` | Number of attention heads |
| `n_positions` | `2048` | Maximum sequence length (positional embeddings) |

### Data

| Field | Default | Description |
|---|---|---|
| `max_seq_len` | `2048` | Token sequence cap — truncated to this length |
| `num_bars_choices` | `[4, 8]` | Window sizes sampled during training |
| `min_tracks` | `1` | Minimum tracks per sample |
| `max_tracks` | `4` | Maximum tracks per sample |
| `min_fill_ratio` | `0.75` | Minimum note density required to accept a window |

### Training objective

| Field | Default | Description |
|---|---|---|
| `infill_probability` | `0.75` | Fraction of samples trained with FillIn tokens |
| `infill_bar_fraction` | `0.5` | Max per-cell infill density (drawn from Uniform(0, this)) |
| `mask_apply_probability` | `0.5` | Fraction of samples with `MASK_BAR` applied |
| `mask_mode` | `2` | `MaskMode`: 0 = RANDOM, 1 = STRUCTURED, 2 = MIXED |

### Optimisation

| Field | Default | Description |
|---|---|---|
| `learning_rate` | `1e-4` | Peak learning rate |
| `batch_size` | `16` | Per-GPU batch size |
| `max_steps` | `100000` | Total training steps |
| `warmup_steps` | `1000` | Linear LR warmup steps |
| `precision` | `"fp16"` | `"fp16"`, `"bf16"`, or `"fp32"` |

### Infrastructure

| Field | Default | Description |
|---|---|---|
| `num_workers` | `0` | **Must be 0** — the C++ MIDI parser is not fork-safe |
| `save_steps` | `5000` | Save a checkpoint every N steps |
| `eval_steps` | `1000` | Run validation every N steps |
| `logger` | `"none"` | `"tensorboard"`, `"wandb"`, or `"none"` |
| `output_dir` | `"checkpoints"` | Where to write checkpoints and the final bundle |

---

## Checkpoint format

By default, training output is saved as a `.safetensors` bundle (`format_version: 2`) containing the weights and metadata:

* **Weights:** Stored natively in SafeTensors format.
* **Metadata:** Inside the SafeTensors file header, the following string keys are defined:
  * `format_version`: `"2"`
  * `arch`: `"gpt2"`
  * `config`: A JSON string representing the model architecture configuration (e.g., `n_embd`, `n_layer`, `n_head`).
  * `encoder_config`: A JSON string representing the full encoder configuration (vocabulary domains, resolution, etc.).

Load with:

```python
from midigpt.inference import InferenceEngine
engine = InferenceEngine.from_checkpoint("checkpoints/run_001/model_final.safetensors")
```

The loader also remains backwards-compatible with:
* Legacy `.pt` packed-bundle files (`format_version: 1`) containing a pickled dict of weights, architecture, and encoder configuration.
* A directory containing `config.json` + `model.pt` (legacy TorchScript representation).
