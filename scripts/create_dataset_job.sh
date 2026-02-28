#!/usr/bin/env bash
# =============================================================================
# create_dataset_job.sh — SLURM batch script for MIDI-GPT dataset creation
#
# Do NOT run this directly. Submit via scripts/create_dataset.sh which
# passes all variables via --export.
# =============================================================================
#SBATCH --nodes=1
# (cpus-per-task, mem, time, account, job-name, output set by the wrapper)

set -euo pipefail

echo "=== MIDI-GPT Dataset Job: $(date) ==="
echo "  Host     : $(hostname)"
echo "  SLURM job: ${SLURM_JOB_ID:-n/a}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK:-?}"
echo ""

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 gcc/12 python/3.11 cmake protobuf

source "$VENV/bin/activate"
python3 -c "import midigpt; print('midigpt loaded OK')"

# ── Build command ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

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

# Input source: tar archive or directory tree
if [[ -n "${TAR_FILE:-}" ]]; then
    CMD+=(--tar "$TAR_FILE" --split "${SPLIT:-0.8,0.1,0.1}")
else
    CMD+=(--data_dir "$DATA_DIR")
fi

# Metadata CSV (optional)
if [[ -n "${METADATA:-}" && -f "$METADATA" ]]; then
    CMD+=(--metadata "$METADATA")
fi

if [[ "$EXPRESSIVE" == "true" ]]; then
    CMD+=(--expressive)
fi

if [[ "$MAX_SIZE" -gt 0 ]]; then
    CMD+=(--max_size "$MAX_SIZE")
fi

if [[ -n "${GENRE_DATA:-}" && -f "$GENRE_DATA" ]]; then
    CMD+=(--genre_data "$GENRE_DATA")
fi

if [[ -n "${SPOTIFY_DATA:-}" && -f "$SPOTIFY_DATA" ]]; then
    CMD+=(--spotify_data "$SPOTIFY_DATA")
fi

if [[ -n "${TENSION_DATA:-}" && -f "$TENSION_DATA" ]]; then
    CMD+=(--tension_data "$TENSION_DATA")
fi

# ── Log config ────────────────────────────────────────────────────────────────
echo "Command: ${CMD[*]}"
echo ""
echo "  Git commit : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Python     : $(python3 --version)"
echo "  torch      : $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "=== Done: $(date) ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.arr 2>/dev/null || echo "(no .arr files yet — check logs above)"
