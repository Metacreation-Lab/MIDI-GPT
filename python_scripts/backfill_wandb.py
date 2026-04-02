"""
Backfill historical TensorBoard data into W&B.

Reads the TensorBoard event file from the first training run (before W&B was
added) and logs all metrics to a new W&B run using the actual global step as
the step counter.  Once done, open the W&B project and overlay this run with
the live run — set the x-axis to `train/global_step` for a seamless curve.

Usage (login node is fine — no GPU needed):
    module load StdEnv/2023 python/3.11.5
    source /scratch/triana24/.venvs/midigpt/bin/activate
    python python_scripts/backfill_wandb.py
"""

import os
import sys

# ── config ──────────────────────────────────────────────────────────────────
TB_DIR = (
    "/scratch/triana24/.midigpt/runs/checkpoints/"
    "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER_gpt2_yellow_metamidi"
    "_Mar_24_02_48_num_bars_8_12_GIT_HASH_"
    "f0010673b34fb4687c26263221a3760b88e11b41/"
    "runs/Mar24_02-48-27_fc10407"
)

# Only upload steps before the W&B run took over (exclusive upper bound).
# Set to None to upload everything in the file.
MAX_STEP = 75000

WANDB_PROJECT = "midi-gpt"
WANDB_ENTITY = "metacreation-lab"
RUN_NAME = "yellow_metamidi_history_0-75k"
# ────────────────────────────────────────────────────────────────────────────

# Load API key from .env if not already in environment
if "WANDB_API_KEY" not in os.environ:
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line.startswith("WANDB_API_KEY="):
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1]
                break

import wandb
from tensorboard.backend.event_processing import event_accumulator

print(f"Loading TensorBoard events from:\n  {TB_DIR}")
ea = event_accumulator.EventAccumulator(TB_DIR)
ea.Reload()

scalar_tags = ea.Tags().get("scalars", [])
print(f"Scalar tags found: {scalar_tags}")

# Build a dict: step -> {tag: value}
step_data: dict[int, dict] = {}
for tag in scalar_tags:
    for event in ea.Scalars(tag):
        if MAX_STEP is not None and event.step > MAX_STEP:
            continue
        step_data.setdefault(event.step, {})[tag] = event.value

if not step_data:
    print("No data to upload — check TB_DIR and MAX_STEP.")
    sys.exit(1)

sorted_steps = sorted(step_data)
print(f"Steps to upload: {sorted_steps[0]} – {sorted_steps[-1]}  ({len(sorted_steps)} points)")

run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=RUN_NAME,
    config={
        "source": "tensorboard_backfill",
        "tb_dir": TB_DIR,
        "original_run": "yellow_metamidi_Mar_24_02_48",
        "step_range": f"{sorted_steps[0]}-{sorted_steps[-1]}",
    },
    tags=["backfill", "yellow_metamidi"],
)

print(f"W&B run created: {run.url}")
print("Uploading...")

for i, step in enumerate(sorted_steps):
    log_dict = dict(step_data[step])
    # Include the global step as a loggable metric so both runs share a
    # common x-axis key in the W&B dashboard.
    log_dict["train/global_step"] = step
    wandb.log(log_dict, step=step)

    if (i + 1) % 100 == 0 or i == len(sorted_steps) - 1:
        print(f"  uploaded {i+1}/{len(sorted_steps)} steps (global step {step})")

wandb.finish()
print("Done. In the W&B dashboard, overlay this run with 'mi91s8ow' and set x-axis → train/global_step.")
