import sys, os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")

from midigpt_compat import midigpt
import json
import random

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
  
  # Get protobuf format for sampling (this is what works correctly)
  protobuf_json = e.midi_to_json_protobuf(midi_input)
  protobuf_data = json.loads(protobuf_json)
  
  # Create status that matches actual track structure
  actual_tracks = protobuf_data.get('tracks', [])
  
  valid_status = {
    'tracks': []
  }
  
  # Configure for each actual track in the MIDI
  for i in range(len(actual_tracks)):
    track_config = {
      'track_id': i,
      'temperature': 0.5,
      'instrument': 'acoustic_grand_piano', 
      'density': 10, 
      'track_type': 10, 
      'ignore': False, 
      'selected_bars': [False, False, True, False], 
      'min_polyphony_q': 'POLYPHONY_ANY', 
      'max_polyphony_q': 'POLYPHONY_ANY', 
      'autoregressive': False,
      'polyphony_hard_limit': 9 
    }
    valid_status['tracks'].append(track_config)

  parami={
          'tracks_per_step': 1, 
          'bars_per_step': 1, 
          'model_dim': 4, 
          'percentage': 100, 
          'batch_size': 1, 
          'temperature': 1.0, 
          'max_steps': 200, 
          'polyphony_hard_limit': 6, 
          'shuffle': True, 
          'verbose': False,  # Turn off verbose to reduce output
          'ckpt': ckpt,
          'sampling_seed': -1,
          'mask_top_k': 0
        }

  # Use protobuf format directly for sampling
  piece = protobuf_json
  status = json.dumps(valid_status)
  param = json.dumps(parami)
  callbacks = midigpt.CallbackManager()
  max_attempts = 3
  
  print(f"Processing {len(actual_tracks)} track(s) from {midi_input}")
  
  midi_str = midigpt.sample_multi_step(piece, status, param, max_attempts, callbacks)
  midi_str = midi_str[0]
  midi_json = json.loads(midi_str)

  # Verify track count
  result_tracks = midi_json.get('tracks', [])
  print(f"Generated {len(result_tracks)} track(s) (input had {len(actual_tracks)})")
  
  if len(result_tracks) != len(actual_tracks):
    print(f"WARNING: Track count changed during generation!")

  e = midigpt.ExpressiveEncoder()
  e.json_to_midi(midi_str, midi_dest)
  
  print(f"Generated MIDI saved to: {midi_dest}")