#!/usr/bin/env bash
# =============================================================================
# hp_search.sh — Submit an Optuna hyperparameter search job to SLURM
#
# Usage:
#   ./scripts/hp_search.sh --dataset /path/to/data.arr --config config/gpt2.json
#
# Required flags:
#   --dataset PATH      Path to .arr dataset file
#   --config  PATH      Path to model config JSON
#
# Optional flags:
#   --encoding STR      Encoder type (default: EXPRESSIVE_ENCODER)
#   --arch STR          Model architecture (default: gpt2)
#   --ngpu INT          Number of GPUs, 1-4 (default: 1)
#   --n_trials INT      Number of Optuna trials (default: 20)
#   --study_name STR    Optuna study name (default: midigpt_hp_search)
#   --storage STR       Optuna storage URL (default: sqlite:///hp_search.db)
#   --per_device_batch_size INT  Per-device micro-batch (default: 16)
#   --label STR         Run label (default: hp_search)
#   --time HH:MM:SS     Walltime (default: 24:00:00)
#   --mem STR           Memory (default: 64G)
#   --account STR       SLURM account (default: $SBATCH_ACCOUNT)
#   --expressive        Enable expressive encoder features
#   --dry-run           Print sbatch command without submitting
#   --extra "..."       Extra args passed verbatim to hp_search.py
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
N_TRIALS=20
STUDY_NAME="midigpt_hp_search"
STORAGE="sqlite:///hp_search.db"
PER_DEVICE_BATCH_SIZE=16
LABEL="hp_search"
WALLTIME="24:00:00"
MEM="64G"
ACCOUNT="${SBATCH_ACCOUNT:-def-pasquier}"
VENV="${VENV:-/scratch/triana24/.venvs/midigpt}"
OUTPUT_DIR="${SCRATCH}/.midigpt/hp_search"
EXPRESSIVE="false"
DRY_RUN=false
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2";               shift 2 ;;
        --config)       CONFIG="$2";                shift 2 ;;
        --encoding)     ENCODING="$2";              shift 2 ;;
        --arch)         ARCH="$2";                  shift 2 ;;
        --ngpu)         NGPU="$2";                  shift 2 ;;
        --n_trials)     N_TRIALS="$2";              shift 2 ;;
        --study_name)   STUDY_NAME="$2";            shift 2 ;;
        --storage)      STORAGE="$2";               shift 2 ;;
        --per_device_batch_size) PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
        --label)        LABEL="$2";                 shift 2 ;;
        --time)         WALLTIME="$2";              shift 2 ;;
        --mem)          MEM="$2";                   shift 2 ;;
        --account)      ACCOUNT="$2";               shift 2 ;;
        --venv)         VENV="$2";                  shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";            shift 2 ;;
        --expressive)   EXPRESSIVE="true";          shift ;;
        --dry-run)      DRY_RUN=true;               shift ;;
        --extra)        EXTRA_ARGS="$2";            shift 2 ;;
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
JOB_NAME="midigpt-hp-${ARCH}-${LABEL}"
LOG_FILE="${SCRATCH}/.midigpt/logs/${JOB_NAME}-%j.out"
mkdir -p "$(dirname "$LOG_FILE")"

# ── Build job script inline ──────────────────────────────────────────────────
JOB_SCRIPT=$(cat <<'JOBEOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
set -euo pipefail

echo "=== MIDI-GPT HP Search: $(date) ==="
echo "  Host      : $(hostname)"
echo "  SLURM job : ${SLURM_JOB_ID:-n/a}"
echo "  GPUs      : ${NGPU}"
echo ""

module purge
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4

source "$VENV/bin/activate"
python3 -c "import midigpt; print('midigpt loaded OK')"
python3 -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.cuda.is_available()}  devices {torch.cuda.device_count()}')"

mkdir -p "$OUTPUT_DIR"

HP_SCRIPT="$REPO_DIR/python_scripts/hp_search.py"

COMMON_ARGS=(
    --arch        "$ARCH"
    --config      "$CONFIG"
    --encoding    "$ENCODING"
    --dataset     "$DATASET"
    --ngpu        "$NGPU"
    --n_trials    "$N_TRIALS"
    --study_name  "$STUDY_NAME"
    --storage     "$STORAGE"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --label       "$LABEL"
    --output      "$OUTPUT_DIR"
)

if [[ "${EXPRESSIVE:-false}" == "true" ]]; then
    COMMON_ARGS+=(--expressive)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
    COMMON_ARGS+=("${EXTRA_ARRAY[@]}")
fi

if [[ "$NGPU" -gt 1 ]]; then
    echo "Launching with torchrun (nproc_per_node=$NGPU)"
    CMD=(torchrun --nproc_per_node "$NGPU" "$HP_SCRIPT" "${COMMON_ARGS[@]}")
else
    echo "Launching with python (single GPU)"
    CMD=(python3 "$HP_SCRIPT" "${COMMON_ARGS[@]}")
fi

echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "=== Done: $(date) ==="
JOBEOF
)

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
    --export=ALL,REPO_DIR="$REPO_DIR",VENV="$VENV",DATASET="$DATASET",CONFIG="$CONFIG",ENCODING="$ENCODING",ARCH="$ARCH",NGPU="$NGPU",N_TRIALS="$N_TRIALS",STUDY_NAME="$STUDY_NAME",STORAGE="$STORAGE",PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE",LABEL="$LABEL",OUTPUT_DIR="$OUTPUT_DIR",EXPRESSIVE="$EXPRESSIVE",EXTRA_ARGS="$EXTRA_ARGS"
)

echo "=== MIDI-GPT Hyperparameter Search Job ==="
echo "  Dataset    : $DATASET"
echo "  Config     : $CONFIG"
echo "  Arch       : $ARCH"
echo "  Encoding   : $ENCODING"
echo "  GPUs       : $NGPU"
echo "  Trials     : $N_TRIALS"
echo "  Study      : $STUDY_NAME"
echo "  Storage    : $STORAGE"
echo "  Micro-batch: $PER_DEVICE_BATCH_SIZE"
echo "  Label      : $LABEL"
echo "  Walltime   : $WALLTIME"
echo "  Memory     : $MEM"
echo "  Account    : $ACCOUNT"
echo "  Output     : $OUTPUT_DIR"
echo "  Log file   : $LOG_FILE"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would execute:"
    echo "  ${SBATCH_CMD[*]} <<< <inline job script>"
    echo ""
    echo "Job script:"
    echo "$JOB_SCRIPT"
else
    echo "$JOB_SCRIPT" | "${SBATCH_CMD[@]}"
fi
