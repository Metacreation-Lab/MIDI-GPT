#!/usr/bin/env bash
# =============================================================================
# benchmark_speed_job.sh — SLURM CPU job for model speed benchmarking
#
# Submit via:
#   sbatch --export=ALL,MODEL_A="...",MODEL_B="..." scripts/benchmark_speed_job.sh
# =============================================================================
#SBATCH --job-name=midigpt-speed-bench
#SBATCH --account=def-pasquier
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR="/scratch/triana24/MIDI-GPT"

module purge
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4

VENV="${VENV:-/scratch/triana24/.venvs/midigpt}"
source "$VENV/bin/activate"

python3 -c "import midigpt; print('midigpt OK')"

OUTPUT="${OUTPUT:-$SCRATCH/.midigpt/speed_bench_$(date +%Y%m%d_%H%M%S).json}"
mkdir -p "$(dirname "$OUTPUT")"

echo "=== MIDI-GPT Speed Benchmark: $(date) ==="
echo "  Model A : $MODEL_A"
echo "  Model B : $MODEL_B"
echo "  Output  : $OUTPUT"
echo ""

python3 "$REPO_DIR/python_scripts/benchmark_speed.py" \
    --model_a   "$MODEL_A" \
    --model_b   "$MODEL_B" \
    --seq_lens  32 64 128 256 512 1024 \
    --warmup    5 \
    --repeats   20 \
    --device    cpu \
    --output    "$OUTPUT"

echo ""
echo "=== Done: $(date) ==="
