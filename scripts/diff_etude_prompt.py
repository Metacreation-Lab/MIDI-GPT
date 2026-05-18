"""Diff orig vs ref prompt token sequences for the Etude benchmark case.

Replicates the exact setup from test_full_generation_speed:
  Etude No. 4.mid, track=0, window bars 2-5, gen=[3, 4], modelDim=4, barsPerStep=2.
"""
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "midigpt_refactor" / "src" / "python"))

import midigpt
import midigpt_refactor._core as _core
from midigpt_refactor._converters import from_cpp, to_cpp
from midigpt_refactor.attributes import AttributeAnalyzer
import copy

MIDI = ROOT / "midigpt_refactor" / "tests" / "comparison" / "midi" / "6338816_Etude No. 4.mid"
CFG_TXT = (ROOT / "models" / "yellow_config.json").read_text()

cfg   = _core.EncoderConfig.from_json(CFG_TXT)
vocab = _core.Vocabulary(cfg)
ref_enc = _core.Encoder(vocab)
ref_dec = _core.Decoder(vocab)
analyzer = AttributeAnalyzer.yellow_default()

orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()

# Window setup matches test_full_generation_speed
WIN_START, WIN_END = 2, 5
WIN_SIZE = 4
GEN_BARS_GLOBAL = [3, 4]
GEN_BARS_LOCAL  = [b - WIN_START for b in GEN_BARS_GLOBAL]  # [1, 2]
TRACK_ID = 0
N_GEN = 2

# ---- ORIG: build trimmed piece + status + prompt tokens
orig_piece = json.loads(orig_enc.midi_to_json(str(MIDI)))
for t in orig_piece.get("tracks", []):
    t["bars"] = t["bars"][WIN_START:WIN_END + 1]
orig_piece_json_win = json.dumps(orig_piece)

status = json.loads(midigpt.status_from_piece(orig_piece_json_win))
for ti, t in enumerate(status["tracks"]):
    for bi in range(len(t.get("selectedBars", []))):
        t["selectedBars"][bi] = (ti == TRACK_ID and bi in GEN_BARS_LOCAL)

param = {
    "ckpt": "",
    "barsPerStep": N_GEN,
    "modelDim": WIN_SIZE,
    "tracksPerStep": 1,
    "temperature": 1.0,
    "batchSize": 1,
    "percentage": 100,
    "polyphonyHardLimit": 10,
}
import torch
extra = {"metadata.json": ""}
torch.jit.load(str(ROOT / "models" / "yellow.pt"), map_location="cpu", _extra_files=extra)
metadata_json_str = extra["metadata.json"]
orig_prompts = midigpt.get_infill_prompts(
    orig_piece_json_win, json.dumps(status), json.dumps(param), metadata_json_str
)
print(f"orig num steps: {len(orig_prompts)}")

# ---- REF: build trimmed score + step + prompt tokens
full_score_cpp = _core.MidiReader(cfg.resolution).read(str(MIDI))

# Trim to window in C++ Score
trimmed_tracks = []
for t in full_score_cpp.tracks:
    nt = _core.Track()
    nt.instrument = t.instrument
    nt.type = t.type
    nt.bars = t.bars[WIN_START:WIN_END + 1]
    nt.attributes = t.attributes
    trimmed_tracks.append(nt)
trimmed = _core.Score()
trimmed.resolution = full_score_cpp.resolution
trimmed.tempo      = full_score_cpp.tempo
trimmed.tracks     = trimmed_tracks
trimmed.notes      = full_score_cpp.notes

# Build selection mask
n_tracks = len(trimmed.tracks)
n_bars   = max((len(t.bars) for t in trimmed.tracks), default=0)
mask = _core.SelectionMask()
sel  = [[False] * n_bars for _ in range(n_tracks)]
for bi in GEN_BARS_LOCAL:
    sel[TRACK_ID][bi] = True
mask.selected       = sel
mask.autoregressive = [False] * n_tracks
mask.ignore         = [False] * n_tracks

# Compute attributes on C++ score
computed_attrs = {}
for t_idx, cpp_track in enumerate(trimmed.tracks):
    attrs = dict(cpp_track.attributes)
    attrs.update(analyzer.compute_track_tokens(trimmed, t_idx))
    for b_idx in range(len(cpp_track.bars)):
        for k, v in analyzer.compute_bar_tokens(trimmed, t_idx, b_idx).items():
            attrs[f"bar_{k}_{b_idx}"] = v
    computed_attrs[t_idx] = attrs

py_score = from_cpp(trimmed)
for t_idx, track in enumerate(py_score.tracks):
    track.attributes = computed_attrs[t_idx]

# StepPlanner with model_dim = WIN_SIZE (match orig)
old_md = cfg.model_dim
cfg.model_dim = WIN_SIZE
try:
    planner = _core.StepPlanner(mask, cfg, N_GEN, 1)
    steps = list(planner.plan())
finally:
    cfg.model_dim = old_md

print(f"ref num steps: {len(steps)}")

ref_prompts = []
for step in steps:
    step_score = copy.deepcopy(py_score)
    for t in step_score.tracks:
        a = dict(t.attributes)
        a["num_bars"] = step.end_bar + 1
        t.attributes = a
    state = _core.SessionState(
        to_cpp(step_score), step, vocab,
        _core.ConstraintGraph(), ref_enc, ref_dec,
    )
    ref_prompts.append(list(state.context_tokens()))

# ---- Diff ----
def pretty_ref(t):
    try:
        tt, val = vocab.decode(t)
        return f"{str(tt).split('.')[-1]}:{val}"
    except Exception:
        return f"?:{t}"

def pretty_orig(t):
    try:
        return orig_enc.pretty_token(t)
    except Exception:
        return f"?:{t}"

for i, (op, rp) in enumerate(zip(orig_prompts, ref_prompts)):
    print(f"\n=== Step {i}: orig len={len(op)}  ref len={len(rp)} ===")
    if op == rp:
        print("  IDENTICAL")
        continue
    minl = min(len(op), len(rp))
    diff_count = 0
    for j in range(minl):
        if op[j] != rp[j]:
            print(f"  [{j:4d}] orig={op[j]:4d} ({pretty_orig(op[j])})  |  ref={rp[j]:4d} ({pretty_ref(rp[j])})")
            diff_count += 1
            if diff_count >= 30:
                print("  ... (truncated)")
                break
    if len(op) != len(rp):
        print(f"  LENGTH DIFF: orig={len(op)} ref={len(rp)}")

if len(orig_prompts) != len(ref_prompts):
    print(f"\nNUM STEPS DIFF: orig={len(orig_prompts)} ref={len(ref_prompts)}")
