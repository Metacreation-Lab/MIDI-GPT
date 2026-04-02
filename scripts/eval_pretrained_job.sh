#!/usr/bin/env bash
# =============================================================================
# eval_pretrained_job.sh — SLURM batch script for evaluating one .pt model
#
# Do NOT run directly. Submit via scripts/eval_pretrained.sh which passes
# MODEL, NUM_BARS, and OUTPUT via --export.
# =============================================================================
#SBATCH --job-name=midigpt-eval-pretrained
#SBATCH --account=def-pasquier
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR="/scratch/triana24/MIDI-GPT"

module purge
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4

VENV="${VENV:-/scratch/triana24/.venvs/midigpt}"
source "$VENV/bin/activate"

python3 -c "import midigpt; print('midigpt OK')"

DATASET="${DATASET:-$HOME/scratch/.midigpt/datasets/EXPRESSIVE_ENCODER_NUM_BARS=4_RESOLUTION_12.arr}"
OUTPUT="${OUTPUT:-$SCRATCH/.midigpt/eval_$(basename "$MODEL" .pt).json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== MIDI-GPT Pretrained Eval: $(date) ==="
echo "  Model    : $MODEL"
echo "  Num bars : $NUM_BARS"
echo "  Dataset  : $DATASET"
echo "  Output   : $OUTPUT"
echo ""

python3 "$REPO_DIR/python_scripts/eval_pretrained.py" \
    --model     "$MODEL" \
    --num_bars  "$NUM_BARS" \
    --dataset   "$DATASET" \
    --splits    1 2 \
    --num_batches 300 \
    --batch_size  32 \
    --output    "$OUTPUT"

echo ""
echo "=== Done: $(date) ==="
