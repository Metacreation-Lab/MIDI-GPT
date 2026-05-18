"""Run orig and ref generation N times with different random seeds and average results."""
import json, sys, time, os, statistics
from pathlib import Path

os.environ["MIDIGPT_VERBOSITY"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "midigpt_refactor" / "src" / "python"))

import torch
import midigpt
import midigpt_refactor._core as _core
import random
from midigpt_refactor._converters import from_cpp, to_cpp
from midigpt_refactor.attributes import AttributeAnalyzer
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.inference.engine import InferenceEngine
from midigpt_refactor.inference.config import GenerationRequest, TrackPrompt, SamplingConfig

MIDI = ROOT / "midigpt_refactor" / "tests" / "comparison" / "midi" / "6338816_Etude No. 4.mid"
MODEL = ROOT / "models" / "yellow.pt"
CFG_TXT = (ROOT / "models" / "yellow_config.json").read_text()

WIN_START, WIN_END = 2, 5
WIN_SIZE = 4
N_GEN    = 2
GEN_BARS_GLOBAL = [3, 4]
GEN_BARS_LOCAL  = [b - WIN_START for b in GEN_BARS_GLOBAL]
TRACK_ID = 0
TEMP = 1.0
N_RUNS = 5
SEEDS = [random.randint(0, 100000) for i in range(N_RUNS)]

cfg     = _core.EncoderConfig.from_json(CFG_TXT)
vocab   = _core.Vocabulary(cfg)
orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
analyzer = AttributeAnalyzer.yellow_default()

print(f"Etude window={WIN_START}-{WIN_END} gen={GEN_BARS_GLOBAL} bps={N_GEN} modelDim={WIN_SIZE} temp={TEMP}")
print(f"Running {N_RUNS} reps each with seeds {SEEDS}\n")

# ---- ORIG: build piece + status once ----
orig_piece = json.loads(orig_enc.midi_to_json(str(MIDI)))
for t in orig_piece["tracks"]:
    t["bars"] = t["bars"][WIN_START:WIN_END + 1]
orig_piece_json = json.dumps(orig_piece)
status = json.loads(midigpt.status_from_piece(orig_piece_json))
for ti, t in enumerate(status["tracks"]):
    for bi in range(len(t.get("selectedBars", []))):
        t["selectedBars"][bi] = (ti == TRACK_ID and bi in GEN_BARS_LOCAL)
param = {
    "ckpt": str(MODEL), "barsPerStep": N_GEN, "modelDim": WIN_SIZE,
    "tracksPerStep": 1, "temperature": TEMP, "batchSize": 1,
    "percentage": 100, "polyphonyHardLimit": 10,
    "sampling_seed": 0,  # placeholder, overwritten per run
}

# ---- REF: setup once ----
_model = torch.jit.load(str(MODEL), map_location="cpu")
_model.eval()
tokenizer = Tokenizer(cfg, analyzer)
engine = InferenceEngine(model=_model, tokenizer=tokenizer, analyzer=analyzer)
engine.warmup()

full_score_cpp = _core.MidiReader(cfg.resolution).read(str(MIDI))
score_prompt_base = from_cpp(full_score_cpp)
for t in score_prompt_base.tracks:
    t.bars = t.bars[WIN_START:WIN_END + 1]

# ---- run loop ----
orig_results = []  # list of (n_tokens, n_notes, wall_ms)
ref_results  = []

for run_i, seed in enumerate(SEEDS):
    # ORIG
    param["sampling_seed"] = seed
    rec = midigpt.RecordTokenSequenceCallback()
    cm  = midigpt.CallbackManager()
    cm.add_callback(rec)
    t0 = time.perf_counter()
    result_str, _, tmg = midigpt.sample_multi_step_timed(
        orig_piece_json, json.dumps(status), json.dumps(param), 1, cm
    )
    orig_wall = (time.perf_counter() - t0) * 1000
    orig_toks = len(rec.tokens)
    rp = json.loads(result_str)
    # bar.events lists note-on AND note-off as separate entries (indices into rp["events"]);
    # count only note-ons (velocity > 0) for parity with ref's note count.
    all_ev = rp.get("events", [])
    orig_notes = sum(
        sum(1 for ei in rp["tracks"][TRACK_ID]["bars"][bi].get("events", []) if all_ev[ei]["velocity"] > 0)
        for bi in GEN_BARS_LOCAL
    )
    orig_results.append((orig_toks, orig_notes, orig_wall))

    # REF
    torch.manual_seed(seed)
    from copy import deepcopy
    score_prompt = deepcopy(score_prompt_base)
    request = GenerationRequest(
        tracks=[TrackPrompt(id=TRACK_ID, bars=GEN_BARS_LOCAL)],
        config=SamplingConfig(
            max_attempts=1, silence_check=False, novelty_check=False,
            bars_per_step=N_GEN, model_dim=WIN_SIZE, temperature=TEMP,
        ),
    )
    session = engine.session(score_prompt, request)
    t0 = time.perf_counter()
    result = session.run()
    ref_wall = (time.perf_counter() - t0) * 1000
    ref_toks = session.gen_count
    ref_notes = sum(len(result.tracks[TRACK_ID].bars[bi].notes) for bi in GEN_BARS_LOCAL)
    ref_results.append((ref_toks, ref_notes, ref_wall))

    print(f"  run {run_i+1} seed={seed:5d}: "
          f"orig toks={orig_toks:4d} notes={orig_notes:4d} wall={orig_wall:6.0f}ms  |  "
          f"ref toks={ref_toks:4d} notes={ref_notes:4d} wall={ref_wall:6.0f}ms")

def stats(xs, idx):
    vals = [r[idx] for r in xs]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0, min(vals), max(vals)

print("\n" + "=" * 80)
print(f"AVERAGED OVER {N_RUNS} RUNS")
print("=" * 80)
print(f"{'metric':<20} {'orig mean±sd':>20} {'orig min..max':>16} {'ref mean±sd':>20} {'ref min..max':>16}")
for name, idx in [("tokens", 0), ("notes", 1), ("wall (ms)", 2)]:
    om, os_, omin, omax = stats(orig_results, idx)
    rm, rs_, rmin, rmax = stats(ref_results, idx)
    print(f"{name:<20} {om:>10.1f}±{os_:<8.1f} {omin:>5.0f}..{omax:<8.0f} {rm:>10.1f}±{rs_:<8.1f} {rmin:>5.0f}..{rmax:<8.0f}")
