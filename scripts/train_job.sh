#!/usr/bin/env bash
# =============================================================================
# train_job.sh — SLURM batch script for MIDI-GPT training
#
# Do NOT run this directly. Submit via scripts/train.sh which passes all
# variables via --export.
# =============================================================================
#SBATCH --nodes=1
# (gpus-per-node, cpus-per-task, mem, time, account, job-name, output set by wrapper)

set -euo pipefail

echo "=== MIDI-GPT Training Job: $(date) ==="
echo "  Host      : $(hostname)"
echo "  SLURM job : ${SLURM_JOB_ID:-n/a}"
echo "  GPUs      : ${NGPU}"
echo "  CPUs      : ${SLURM_CPUS_PER_TASK:-?}"
echo ""

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4

source "$VENV/bin/activate"
python3 -c "import midigpt; print('midigpt loaded OK')"
python3 -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.cuda.is_available()}  devices {torch.cuda.device_count()}')"

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/midigpt/runs}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/midigpt/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Log config ───────────────────────────────────────────────────────────────
echo "  Git commit : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Python     : $(python3 --version)"
echo "  torch      : $(python3 -c 'import torch; print(torch.__version__)')"
echo "  Dataset    : $DATASET"
echo "  Encoding   : $ENCODING"
echo "  Arch       : $ARCH"
echo "  Config     : $CONFIG"
echo "  Batch size : $BATCH_SIZE"
echo "  Num bars   : $NUM_BARS"
echo "  Label      : $LABEL"
echo ""

# ── Build training command ───────────────────────────────────────────────────
TRAIN_SCRIPT="$REPO_DIR/python_scripts/train.py"

COMMON_ARGS=(
    --arch        "$ARCH"
    --config      "$CONFIG"
    --encoding    "$ENCODING"
    --dataset     "$DATASET"
    --ngpu        "$NGPU"
    --batch_size  "$BATCH_SIZE"
    --num_bars    "$NUM_BARS"
    --label       "$LABEL"
    --output      "$OUTPUT_DIR"
    --log_dir     "$LOG_DIR"
)

# Optional: checkpoint resume
if [[ -n "${CKPT:-}" ]]; then
    COMMON_ARGS+=(--ckpt "$CKPT")
    if [[ -n "${CKPT_NUM:-}" ]]; then
        COMMON_ARGS+=(--ckpt_num "$CKPT_NUM")
    fi
fi

# Optional: expressive encoder
if [[ "${EXPRESSIVE:-false}" == "true" ]]; then
    COMMON_ARGS+=(--expressive)
fi

# Optional: extra args passed through from wrapper
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
    COMMON_ARGS+=("${EXTRA_ARRAY[@]}")
fi

# ── Launch ───────────────────────────────────────────────────────────────────
if [[ "$NGPU" -gt 1 ]]; then
    echo "Launching with torchrun (nproc_per_node=$NGPU)"
    CMD=(
        torchrun
        --nproc_per_node "$NGPU"
        "$TRAIN_SCRIPT"
        "${COMMON_ARGS[@]}"
    )
else
    echo "Launching with python (single GPU)"
    CMD=(
        python3
        "$TRAIN_SCRIPT"
        "${COMMON_ARGS[@]}"
    )
fi

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=== Done: $(date) ==="
