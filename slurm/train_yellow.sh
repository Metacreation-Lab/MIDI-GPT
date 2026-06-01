#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Train yellow encoder (default GPT-2) on full GigaMIDI v2.0.0 — 3 days
#
# Usage:
#   sbatch slurm/train_yellow.sh              # full H100 (80 GB)
#   sbatch --gres=gpu:a100_4g.40gb:1 \
#          --export=ALL,GPU_TIER=40gb \
#          slurm/train_yellow.sh              # A100 40 GB MIG slice
#
# Before submitting, download GigaMIDI and run preprocess on a login/CPU node:
#   python -m midigpt.training.preprocess \
#       --parquet "$SCRATCH/MIDI-GPT/data/v2.0.0/train/*.parquet" \
#       --encoder-config models/yellow_encoder.json
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=def-pasquier
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --job-name=midigpt-yellow

# ── environment ───────────────────────────────────────────────────────────────

module purge
module load arrow/19.0.1

PROJECT="$HOME/projects/def-pasquier/$USER/MIDI-GPT"
VENV="$HOME/scratch/MIDI-GPT/.venv"
DATA_DIR="$SCRATCH/MIDI-GPT/data/v2.0.0"
RUN_ID="yellow-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$SCRATCH/MIDI-GPT/runs/$RUN_ID"

# Select config based on GPU tier (override with --export=ALL,GPU_TIER=40gb)
GPU_TIER="${GPU_TIER:-h100}"
CONFIG="$PROJECT/slurm/configs/yellow_${GPU_TIER}.json"

source "$VENV/bin/activate"
cd "$PROJECT"

echo "Run : $RUN_ID"
echo "Config : $CONFIG"
echo "Data : $DATA_DIR/train/*.parquet"
echo "Output : $OUTPUT_DIR"
echo "GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── download GigaMIDI if not present ─────────────────────────────────────────

if [ ! -d "$DATA_DIR/train" ]; then
    echo "Downloading GigaMIDI v2.0.0 …"
    python3 - << PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Metacreation/GigaMIDI",
    repo_type="dataset",
    revision="2.0.0",
    local_dir=os.environ["DATA_DIR"],
    ignore_patterns=["*.md", "*.txt"],
)
PYEOF
fi

# ── preprocess (builds valid-index cache if missing; no-ops on cache hit) ─────

echo "Preprocessing shards …"
python3 -m midigpt.training.preprocess \
    --parquet "$DATA_DIR/train/*.parquet" \
    --encoder-config models/yellow_encoder.json

# ── train ─────────────────────────────────────────────────────────────────────

echo ""
echo "Starting training …"
python3 -m midigpt.training.trainer \
    --config      "$CONFIG" \
    --train-data  "$DATA_DIR/train/*.parquet" \
    --output-dir  "$OUTPUT_DIR"

echo ""
echo "Done. Output: $OUTPUT_DIR"
