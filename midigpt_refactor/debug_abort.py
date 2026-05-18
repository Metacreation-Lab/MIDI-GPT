import sys
from pathlib import Path
import midigpt
import os
import contextlib

@contextlib.contextmanager
def silence_stdio():
    fd_out = 1
    fd_err = 2
    saved_out = os.dup(fd_out)
    saved_err = os.dup(fd_err)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd_out)
    os.dup2(devnull, fd_err)
    try:
        yield
    finally:
        os.dup2(saved_out, fd_out); os.dup2(saved_err, fd_err)
        os.close(saved_out); os.close(saved_err); os.close(devnull)

def test_files():
    enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
    midi_dir = Path("tests/comparison/midi")
    for mp in sorted(midi_dir.glob("*.mid")):
        print(f"Testing {mp.name}...")
        sys.stdout.flush()
        try:
            with silence_stdio():
                pj = enc.midi_to_json(str(mp))
                tokens = enc.json_to_tokens(pj)
            print(f"  OK (tokens: {len(tokens)})")
        except Exception as e:
            print(f"  Failed with Exception: {e}")

if __name__ == "__main__":
    test_files()