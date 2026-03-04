#!/usr/bin/env bash
# =============================================================================
# train.sh — Submit a MIDI-GPT training job to SLURM
#
# Usage:
#   ./scripts/train.sh --dataset /path/to/data.arr --config config/gpt2.json
#
# Required flags:
#   --dataset PATH      Path to .arr dataset file
#   --config  PATH      Path to model config JSON
#
# Optional flags:
#   --encoding STR      Encoder type (default: EXPRESSIVE_ENCODER)
#   --arch STR          Model architecture (default: gpt2)
#   --ngpu INT          Number of GPUs, 1-4 (default: 1)
#   --batch_size INT    Total batch size (default: 32)
#   --num_bars INT      Bars per sample (default: 4)
#   --label STR         Run label (default: v1)
#   --time HH:MM:SS     Walltime (default: 12:00:00)
#   --mem STR           Memory (default: 64G)
#   --account STR       SLURM account (default: $SBATCH_ACCOUNT)
#   --output_dir PATH   Checkpoint output dir
#   --log_dir PATH      Tensorboard log dir
#   --ckpt STR          Checkpoint name to resume from
#   --ckpt_num INT      Checkpoint step number to resume from
#   --expressive        Enable expressive encoder features
#   --dry-run           Print sbatch command without submitting
#   --extra "..."       Extra args passed verbatim to train.py
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET="/home/triana24/scratch/.midigpt/datasets/EXPRESSIVE_ENCODER_NUM_BARS=4_RESOLUTION_12.arr"
CONFIG="/home/triana24/scratch/MIDI-GPT/python_scripts/config/gpt2.json"
ENCODING="EXPRESSIVE_ENCODER"
ARCH="gpt2"
NGPU=1
BATCH_SIZE=64
NUM_BARS=4
LABEL="v5"
WALLTIME="24:00:00"
MEM="64G"
ACCOUNT="${SBATCH_ACCOUNT:-def-pasquier}"
VENV="${VENV:-/scratch/triana24/.venvs/midigpt}"
OUTPUT_DIR="${SCRATCH}/.midigpt/runs"
LOG_DIR="${SCRATCH}/.midigpt/runs/logs"
CKPT="EXPRESSIVE_ENCODER_gpt2_v5_Mar_01_13_52_num_bars_4_12_GIT_HASH_4e3ae1af74d0ae1026f9f7b51fc0f50edb6e84c4"
CKPT_NUM="85000"
EXPRESSIVE="false"
DRY_RUN=false
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2";       shift 2 ;;
        --config)       CONFIG="$2";        shift 2 ;;
        --encoding)     ENCODING="$2";      shift 2 ;;
        --arch)         ARCH="$2";          shift 2 ;;
        --ngpu)         NGPU="$2";          shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";    shift 2 ;;
        --num_bars)     NUM_BARS="$2";      shift 2 ;;
        --label)        LABEL="$2";         shift 2 ;;
        --time)         WALLTIME="$2";      shift 2 ;;
        --mem)          MEM="$2";           shift 2 ;;
        --account)      ACCOUNT="$2";       shift 2 ;;
        --venv)         VENV="$2";          shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --log_dir)      LOG_DIR="$2";       shift 2 ;;
        --ckpt)         CKPT="$2";          shift 2 ;;
        --ckpt_num)     CKPT_NUM="$2";      shift 2 ;;
        --expressive)   EXPRESSIVE="true";  shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        --extra)        EXTRA_ARGS="$2";    shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required" >&2
    exit 1
fi
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required" >&2
    exit 1
fi
if [[ ! -f "$DATASET" ]]; then
    echo "Error: dataset not found: $DATASET" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi
if [[ "$NGPU" -lt 1 || "$NGPU" -gt 4 ]]; then
    echo "Error: --ngpu must be 1-4" >&2
    exit 1
fi
if [[ -z "$ACCOUNT" ]]; then
    echo "Error: --account required (or set SBATCH_ACCOUNT)" >&2
    exit 1
fi
if [[ ! -d "$VENV" ]]; then
    echo "Error: venv not found: $VENV" >&2
    exit 1
fi

# ── Compute resources ────────────────────────────────────────────────────────
CPUS_PER_TASK=$((NGPU * 8))
JOB_NAME="midigpt-train-${ARCH}-${LABEL}"
LOG_FILE="${SCRATCH}/.midigpt/logs/${JOB_NAME}-%j.out"
mkdir -p "$(dirname "$LOG_FILE")"

# ── Submit ───────────────────────────────────────────────────────────────────
SBATCH_CMD=(
    sbatch
    --job-name="$JOB_NAME"
    --account="$ACCOUNT"
    --gpus-per-node="h100:${NGPU}"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$WALLTIME"
    --output="$LOG_FILE"
    --export=ALL,REPO_DIR="$REPO_DIR",VENV="$VENV",DATASET="$DATASET",CONFIG="$CONFIG",ENCODING="$ENCODING",ARCH="$ARCH",NGPU="$NGPU",BATCH_SIZE="$BATCH_SIZE",NUM_BARS="$NUM_BARS",LABEL="$LABEL",OUTPUT_DIR="$OUTPUT_DIR",LOG_DIR="$LOG_DIR",CKPT="$CKPT",CKPT_NUM="$CKPT_NUM",EXPRESSIVE="$EXPRESSIVE",EXTRA_ARGS="$EXTRA_ARGS"
    "$SCRIPT_DIR/train_job.sh"
)

echo "=== MIDI-GPT Training Job ==="
echo "  Dataset  : $DATASET"
echo "  Config   : $CONFIG"
echo "  Arch     : $ARCH"
echo "  Encoding : $ENCODING"
echo "  GPUs     : $NGPU"
echo "  Batch    : $BATCH_SIZE"
echo "  Bars     : $NUM_BARS"
echo "  Label    : $LABEL"
echo "  Walltime : $WALLTIME"
echo "  Memory   : $MEM"
echo "  Account  : $ACCOUNT"
echo "  Log file : $LOG_FILE"
if [[ -n "$CKPT" ]]; then
    echo "  Resume   : $CKPT (step $CKPT_NUM)"
fi
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would execute:"
    echo "  ${SBATCH_CMD[*]}"
else
    "${SBATCH_CMD[@]}"
fi
