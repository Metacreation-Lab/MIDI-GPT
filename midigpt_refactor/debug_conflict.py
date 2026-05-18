import midigpt
import midigpt_refactor._core as _core

enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
print("Calling orig_enc.midi_to_json...")
pj = enc.midi_to_json("tests/comparison/midi/6338816_Etude No. 4.mid")
print("Calling orig_enc.json_to_tokens...")
try:
    tokens = enc.json_to_tokens(pj)
    print("Success")
except Exception as e:
    print(f"Caught exception: {e}")
print("Done!")
