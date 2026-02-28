#!/usr/bin/env bash
# =============================================================================
# create_dataset.sh — Submit (or test) a MIDI-GPT dataset creation job
#
# Usage:
#   bash scripts/create_dataset.sh           # submit SLURM job
#   bash scripts/create_dataset.sh --test    # dry-run on login node
# =============================================================================
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-}"
TAR_FILE="${TAR_FILE:-}"
SPLIT="${SPLIT:-0.9,0.05,0.05}"

METADATA="${METADATA:-}"

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/.midigpt/datasets}"

ENCODING="${ENCODING:-EXPRESSIVE_ENCODER}"
EXPRESSIVE="${EXPRESSIVE:-false}"
DATA_TYPE="${DATA_TYPE:-Drum+Music}"

NUM_BARS="${NUM_BARS:-4}"
RESOLUTION="${RESOLUTION:-12}"
DELTA_RESOLUTION="${DELTA_RESOLUTION:-1920}"
MAX_SIZE="${MAX_SIZE:--1}"
NTHREADS="${NTHREADS:-16}"

GENRE_DATA="${GENRE_DATA:-}"
SPOTIFY_DATA="${SPOTIFY_DATA:-}"
TENSION_DATA="${TENSION_DATA:-}"

# ── SLURM SETTINGS ────────────────────────────────────────────────────────────
SLURM_ACCOUNT="${SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-def-pasquier}}"
SLURM_TIME="${SLURM_TIME:-03:00:00}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-midigpt-dataset}"

VENV="${VENV:-$SCRATCH/.venvs/midigpt}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TEST_MODE=false

for arg in "$@"; do
    case "$arg" in
        --test) TEST_MODE=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ -z "$TAR_FILE" && -z "$DATA_DIR" ]]; then
    echo "ERROR: set either DATA_DIR or TAR_FILE"
    exit 1
fi
if [[ -n "$TAR_FILE" && -n "$DATA_DIR" ]]; then
    echo "ERROR: set only one of DATA_DIR or TAR_FILE"
    exit 1
fi
if [[ -n "$TAR_FILE" && ! -f "$TAR_FILE" ]]; then
    echo "ERROR: TAR_FILE not found: $TAR_FILE"
    exit 1
fi
if [[ -n "$DATA_DIR" && ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR"
    exit 1
fi
if [[ -n "$METADATA" && ! -f "$METADATA" ]]; then
    echo "ERROR: METADATA not found: $METADATA"
    exit 1
fi

# ── Build base command ────────────────────────────────────────────────────────
CMD=(
    python3 "$REPO_DIR/python_scripts/create_dataset.py"
    --output     "$OUTPUT_DIR"
    --encoding   "$ENCODING"
    --type       "$DATA_TYPE"
    --num_bars   "$NUM_BARS"
    --resolution "$RESOLUTION"
    --delta_resolution "$DELTA_RESOLUTION"
    --nthreads   "$NTHREADS"
)

if [[ -n "$TAR_FILE" ]]; then
    CMD+=(--tar "$TAR_FILE" --split "$SPLIT")
else
    CMD+=(--data_dir "$DATA_DIR")
fi

[[ -n "$METADATA" ]]      && CMD+=(--metadata "$METADATA")
[[ "$EXPRESSIVE" == true ]] && CMD+=(--expressive)
[[ "$MAX_SIZE" -gt 0 ]]   && CMD+=(--max_size "$MAX_SIZE")

[[ -n "$GENRE_DATA"   && -f "$GENRE_DATA"   ]] && CMD+=(--genre_data "$GENRE_DATA")
[[ -n "$SPOTIFY_DATA" && -f "$SPOTIFY_DATA" ]] && CMD+=(--spotify_data "$SPOTIFY_DATA")
[[ -n "$TENSION_DATA" && -f "$TENSION_DATA" ]] && CMD+=(--tension_data "$TENSION_DATA")

# ── TEST MODE (login node) ────────────────────────────────────────────────────
if $TEST_MODE; then
    echo ">>> TEST MODE (login node) <<<"
    source "$VENV/bin/activate"
    "${CMD[@]}" --nthreads 1 --max_size 10 --test yes
    echo "Dry-run passed."
    exit 0
fi

# ── SLURM submission ──────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --job-name=$SLURM_JOB_NAME
#SBATCH --time=$SLURM_TIME
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --output=$OUTPUT_DIR/logs/slurm-$SLURM_JOB_NAME.out

set -euo pipefail

echo "=== MIDI-GPT Dataset Job ==="
echo "Host : \$(hostname)"
echo "Job  : \$SLURM_JOB_ID"
echo "CPUs : \$SLURM_CPUS_PER_TASK"
echo ""

module purge
module load StdEnv/2023 gcc/12 python/3.11 cmake protobuf

source "$VENV/bin/activate"
python3 -c "import midigpt; print('midigpt OK')"

echo "Command:"
echo "${CMD[*]}"
echo ""

${CMD[@]}

echo ""
echo "Done."
ls -lh "$OUTPUT_DIR"/*.arr 2>/dev/null || true
EOF

echo "Job submitted. Logs: $OUTPUT_DIR/logs/slurm-<jobid>.out"