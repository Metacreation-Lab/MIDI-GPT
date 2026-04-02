#!/usr/bin/env bash
# =============================================================================
# eval_pretrained_gpu_job.sh — Evaluate one .pt model on a GPU node
#
# Submit via:
#   sbatch --export=ALL,MODEL="...",NUM_BARS=4 scripts/eval_pretrained_gpu_job.sh
# =============================================================================
#SBATCH --job-name=midigpt-eval-pretrained-gpu
#SBATCH --account=def-pasquier
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR="/scratch/triana24/MIDI-GPT"

module purge
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4

VENV="${VENV:-/scratch/triana24/.venvs/midigpt}"
source "$VENV/bin/activate"

python3 -c "import midigpt; print('midigpt OK')"
python3 -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.cuda.is_available()}')"

DATASET="${DATASET:-$HOME/scratch/.midigpt/datasets/EXPRESSIVE_ENCODER_NUM_BARS=4_RESOLUTION_12.arr}"
OUTPUT="${OUTPUT:-$SCRATCH/.midigpt/eval_gpu_$(basename "$MODEL" .pt).json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== MIDI-GPT Pretrained GPU Eval: $(date) ==="
echo "  Model    : $MODEL"
echo "  Num bars : $NUM_BARS"
echo "  Dataset  : $DATASET"
echo "  Output   : $OUTPUT"
echo ""

python3 "$REPO_DIR/python_scripts/eval_pretrained.py" \
    --model       "$MODEL" \
    --num_bars    "$NUM_BARS" \
    --dataset     "$DATASET" \
    --splits      1 2 \
    --num_batches 300 \
    --batch_size  32 \
    --device      cuda \
    --output      "$OUTPUT"

echo ""
echo "=== Done: $(date) ==="
