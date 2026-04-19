#!/bin/bash

# 1. Load required modules (Matched with run_tokenization_test.sh)
echo "Loading standard HPC modules..."
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4 cuda/12.2

# 2. Activate Virtual Environment
echo "Activating virtual environment..."
source /scratch/triana24/.venvs/midigpt/bin/activate

# 3. Setup Simulation Parameters
MODEL_PATH="/home/triana24/scratch/.midigpt/models/ghost_baseline_340k.pt"
NUM_BARS=8
MIDI_PATH="tests/short_midi/generated_${NUM_BARS}bar.mid"

# 4. Generate the source MIDI
echo "Generating ${NUM_BARS}-bar MIDI..."
python tests/short_midi/generate_long_midi.py $NUM_BARS "$MIDI_PATH"

# 5. Run the simulation
# Parameters:
# --buffer 4: Human plays 4 bars before AI starts
# --lookahead 1: AI generates 1 bar ahead of current playhead
# --model_dim 8: Sliding window of 8 bars (Constraint: k+j < model_dim)

echo "Launching Real-Time Simulation..."
python python_scripts_for_testing/simulate_realtime_agent.py \
    --midi "$MIDI_PATH" \
    --ckpt "$MODEL_PATH" \
    --buffer 4 \
    --lookahead 1 \
    --num_anticipated_bars 1 \
    --model_dim 8 \
    --adapt_buffer \
    --delay 0.5 \
    --agent_instrument 1 \
    --output "outputs/realtime_sim_${NUM_BARS}bar.mid"

echo "Simulation Concluded."