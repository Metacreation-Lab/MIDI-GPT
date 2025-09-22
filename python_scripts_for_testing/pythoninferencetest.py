import sys, os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")

# Direct import - no compatibility layer
import midigpt
import json
import random

import sys, os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")

import midigpt
import json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default='')
    args = parser.parse_args()

    ckpt = args.ckpt
    midi_input = args.midi
    if args.out != '':
        midi_dest = args.out
    else:
        midi_dest = os.path.join(os.path.split(midi_input)[0], 'midigpt_gen.mid')
    
    e = midigpt.ExpressiveEncoder()
    midi_json_input = json.loads(e.midi_to_json(midi_input))
    
    # ... [same sampling code as before] ...
    
    # Test both output methods:
    
    print("🎵 Testing multi-track output (standard)...")
    e.json_to_midi(midi_str, "output_multitrack.mid")
    
    print("🎵 Testing single-track output...")
    # Use the json_track_to_midi method with single_track parameter
    e.json_track_to_midi(midi_str, "output_singletrack.mid", 0)  # Just track 0
    
    print("✅ Created both versions:")
    print("   Multi-track: output_multitrack.mid")
    print("   Single-track: output_singletrack.mid")
    print("\n🔍 Compare these in Reaper to see the difference!")