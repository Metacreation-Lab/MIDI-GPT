#!/bin/bash
# One shard of the request matrix on an Alliance GPU node: cluster.sh <model>:<shard>:<shards> [run.py args]
# Expects this source tree at $SCRATCH/midigpt-matrix/src with a STAMP file naming its commit; builds it into
# $SCRATCH/.venvs/midigpt-matrix once per commit (locked, so the shards share one build).
# Results: $SCRATCH/midigpt-matrix/runs/<STAMP>/ (one .jsonl per shard, failures/ to replay).
set -euo pipefail
IFS=: read -r MODEL SHARD SHARDS <<< "$1"; shift
M=$SCRATCH/midigpt-matrix; SRC=$M/src; V=$SCRATCH/.venvs/midigpt-matrix
STAMP=$(cat "$SRC/STAMP")
module load StdEnv/2023 gcc python/3.12 cuda/12.6 cmake arrow
mkdir -p "$SCRATCH/.venvs" "$M/runs/$STAMP"
(   flock 9
    if [ "$(cat "$V/.ok" 2>/dev/null)" != "$STAMP" ]; then
        rm -rf "$V"; python -m venv "$V"; . "$V/bin/activate"
        pip install --upgrade pip
        pip install torch numpy tqdm huggingface_hub safetensors
        pip install "$SRC"
        echo "$STAMP" > "$V/.ok"
    fi
) 9>"$SCRATCH/.venvs/.midigpt-matrix.lock"
. "$V/bin/activate"
export HF_HOME=$SCRATCH/midigpt-eval/hf PYTHONUNBUFFERED=1
cd "$M/runs/$STAMP"     # not the source tree: midigpt must come from the venv
python -c "import torch, midigpt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), midigpt.__file__)"
python "$SRC/scripts/request_matrix/run.py" --data "$SCRATCH/MIDI-GPT/data/v2.0.0/test" --model "$MODEL" \
    --shard "$SHARD" --shards "$SHARDS" --seeds 6 --device cuda --out "$M/runs/$STAMP" "$@"
