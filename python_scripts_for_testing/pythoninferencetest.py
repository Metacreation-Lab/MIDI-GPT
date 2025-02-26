import sys, os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt
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
  midi_json_input = json.loads(e.midi_to_json(midi_input))
  valid_status={'tracks': 
                [
                  {
                    'track_id': 0,
                    'temperature' : 0.5,
                    'instrument': 'acoustic_grand_piano', 
                    'density': 10, 
                    'track_type': 10, 
                    'ignore': False, 
                    'selected_bars': [False, False, True, False ], 
                    'min_polyphony_q': 'POLYPHONY_ANY', 
                    'max_polyphony_q': 'POLYPHONY_ANY', 
                    'autoregressive': False,
                    'polyphony_hard_limit': 9 
                  }
                ]
              }
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
          'verbose': True, 
          'ckpt': ckpt,
          'sampling_seed': -1,
          'mask_top_k': 0
        }

  piece = json.dumps(midi_json_input)
  status = json.dumps(valid_status)
  param = json.dumps(parami)
  callbacks = midigpt.CallbackManager()
  max_attempts = 3
  midi_str = midigpt.sample_multi_step(piece, status, param, max_attempts, callbacks)
  midi_str=midi_str[0]
  midi_json = json.loads(midi_str)

  e = midigpt.ExpressiveEncoder()
  e.json_to_midi(midi_str, midi_dest)
