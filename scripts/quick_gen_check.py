"""
Generation comparison: orig vs refactored OSC server.

Each rep uses a randomly-generated musical context:
  - 1–16 conditioning tracks, random GM instruments
  - Random note density, pitch range, velocity range per track
  - Bar clock gated on generation: next bar sent only after
    both servers return /generated/close (mirrors real Max usage)

Config via env vars:
  MIDIGPT_CKPT          path to yellow.pt
  MIDIGPT_GEN_TIMEOUT   seconds before a generation is abandoned (0 = off)
"""

import os, sys, time, random, threading, math, statistics
from dataclasses import dataclass, field
from pathlib import Path

# Silence C++ stdout trace before importing midigpt
os.environ.setdefault("MIDIGPT_VERBOSITY", "0")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw): return it

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT      = os.environ.get("MIDIGPT_CKPT", str(REPO_ROOT / "models" / "yellow.pt"))
CLIENT    = ("127.0.0.1", 9999)

sys.path.insert(0, str(REPO_ROOT / "src" / "python" / "midigpt"))
sys.path.insert(0, str(REPO_ROOT / "midigpt_refactor" / "src" / "python"))

# ── tunables ──────────────────────────────────────────────────────────────────
MODEL_DIM      = 8
BUFFER_BARS    = 4
LOOKAHEAD      = 1
BARS_PER_STEP  = 1
N_BARS_GEN     = 8    # inference passes per rep (= bars after buffer)
N_REPS         = 1
GEN_TIMEOUT    = 5.0   # seconds; server abandons inference and sends error if exceeded
MAX_ATTEMPTS   = 1
# ─────────────────────────────────────────────────────────────────────────────

ALL_GM = list(range(0, 128))


@dataclass
class TrackConfig:
    instrument:   int
    notes_per_bar:int
    min_pitch:    int
    max_pitch:    int
    min_vel:      int
    max_vel:      int


@dataclass
class RepResult:
    rep:           int
    n_cond_tracks: int
    tracks:        list
    orig_opens:    int = 0
    orig_notes:    int = 0
    orig_errs:     int = 0
    orig_latency:  float = float("nan")
    orig_intervals:list = field(default_factory=list)
    orig_notes_per_bar:list = field(default_factory=list)
    ref_opens:     int = 0
    ref_notes:     int = 0
    ref_errs:      int = 0
    ref_latency:   float = float("nan")
    ref_intervals: list = field(default_factory=list)
    ref_notes_per_bar:list = field(default_factory=list)

    def orig_silence_rate(self):
        if not self.orig_notes_per_bar: return float("nan")
        return sum(1 for n in self.orig_notes_per_bar if n == 0) / len(self.orig_notes_per_bar)

    def ref_silence_rate(self):
        if not self.ref_notes_per_bar: return float("nan")
        return sum(1 for n in self.ref_notes_per_bar if n == 0) / len(self.ref_notes_per_bar)


class Cap:
    def __init__(self, label: str):
        self.label      = label
        self.msgs       = []
        self._lock      = threading.Lock()
        self._gen_event = threading.Event()
        self._open_times:  list = []
        self._close_times: list = []
        self._notes_by_bar: dict = {}
        self._current_bar = -1
        # for per-step verbosity
        self._last_open_t: float = 0.0

    def __call__(self, addr, *args):
        t = time.monotonic()
        with self._lock:
            self.msgs.append((addr, args))
            if addr == "/midigpt/generated/open":
                self._open_times.append(t)
                self._last_open_t = t
                bar = args[1] if len(args) > 1 else -1
                self._current_bar = bar
                self._notes_by_bar.setdefault(bar, 0)
            elif addr == "/midigpt/generated/note":
                bar = args[1] if len(args) > 1 else self._current_bar
                self._notes_by_bar[bar] = self._notes_by_bar.get(bar, 0) + 1
            elif addr == "/midigpt/generated/close":
                self._close_times.append(t)
                elapsed = t - self._last_open_t
                bar = args[1] if len(args) > 1 else self._current_bar
                notes = self._notes_by_bar.get(bar, 0)
                print(f"    [{self.label}] bar {bar:>3}  {notes:>4} notes  "
                      f"({elapsed*1000:.0f}ms)", flush=True)
                self._gen_event.set()
            elif addr == "/midigpt/error":
                code = args[0] if args else "?"
                msg  = args[1] if len(args) > 1 else ""
                print(f"    [{self.label}] ERROR {code}: {msg}", flush=True)
                if code == 5:   # ERR_GENERATION — unblock wait so test moves on
                    self._gen_event.set()

    def wait_for_generated(self, timeout=GEN_TIMEOUT + 2):
        fired = self._gen_event.wait(timeout)
        self._gen_event.clear()
        return fired

    def find(self, prefix):
        with self._lock:
            return [m for m in self.msgs if m[0].startswith(prefix)]

    def errors(self):
        return [m[1][0] for m in self.find("/midigpt/error")]

    def notes_per_bar_list(self):
        with self._lock:
            return [self._notes_by_bar[k] for k in sorted(self._notes_by_bar)]

    def intervals(self):
        ct = self._close_times
        return [ct[i+1] - ct[i] for i in range(len(ct)-1)] if len(ct) > 1 else []

    def clear(self):
        with self._lock:
            self.msgs.clear()
            self._open_times.clear()
            self._close_times.clear()
            self._notes_by_bar.clear()
            self._current_bar = -1
        self._gen_event.clear()


def make_orig(params):
    import midigpt.osc_server as mod
    srv = mod.MidiGPTServer(ckpt=CKPT, listen_port=7400, max_attempts=MAX_ATTEMPTS)
    cap = Cap("orig")
    srv._send = cap
    srv._params.update(params)
    return srv, cap


def build_ref_engine():
    import torch, midigpt_refactor._core as _core
    from midigpt_refactor.inference.engine import InferenceEngine
    from midigpt_refactor.tokenizer.tokenizer import Tokenizer
    from midigpt_refactor.attributes import AttributeAnalyzer
    print("  loading model…", flush=True)
    model = torch.jit.load(CKPT, map_location="cpu")
    model.eval()
    cfg = _core.EncoderConfig.from_json(
        (REPO_ROOT / "models" / "yellow_config.json").read_text())
    engine = InferenceEngine(model, Tokenizer(cfg), AttributeAnalyzer.yellow_default())
    print("  warming up…", flush=True)
    engine.warmup()
    return engine


def make_ref(engine, params):
    from midigpt_refactor.server.osc_server import MidiGPTServer as Ref
    srv = Ref(engine=engine, listen_port=7401, max_attempts=MAX_ATTEMPTS)
    cap = Cap("ref ")
    srv._send = cap
    srv._params.update(params)
    return srv, cap


def random_track_config(rng: random.Random) -> TrackConfig:
    inst      = rng.choice(ALL_GM)
    npb       = rng.randint(1, 16)
    min_pitch = rng.randint(21, 72)
    max_pitch = min(min_pitch + rng.randint(12, 48), 108)
    min_vel   = rng.randint(40, 80)
    max_vel   = min(min_vel + rng.randint(20, 60), 127)
    return TrackConfig(inst, npb, min_pitch, max_pitch, min_vel, max_vel)


def run_session(orig, orig_cap, ref, ref_cap, n_cond, tracks, seed, rep_label) -> RepResult:
    rng = random.Random(seed)
    params = {
        "buffer_bars":          BUFFER_BARS,
        "lookahead_bars":       LOOKAHEAD,
        "num_anticipated_bars": BARS_PER_STEP,
        "model_dim":            MODEL_DIM,
        "gen_timeout":          GEN_TIMEOUT,
    }
    for srv, cap in [(orig, orig_cap), (ref, ref_cap)]:
        srv._state = "UNINITIALIZED"
        srv._piece = None
        srv._params.update(params)
        cap.clear()

    def call(h, *a):
        path = "/midigpt/" + h.replace("handle_", "").replace("_", "/", 1)
        getattr(orig, h)(CLIENT, path, *a)
        getattr(ref,  h)(CLIENT, path, *a)

    def send_bar(bar_idx):
        for tid, tc in enumerate(tracks):
            for ni in range(tc.notes_per_bar):
                call("handle_note", tid, rng.randint(tc.min_pitch, tc.max_pitch),
                     rng.randint(tc.min_vel, tc.max_vel),
                     round(ni / tc.notes_per_bar, 3),
                     round(1.0 / tc.notes_per_bar, 3), bar_idx)
        call("handle_bar_end", bar_idx, 4, 4)

    call("handle_session_init", rep_label)
    for tid, tc in enumerate(tracks):
        call("handle_track_create", tid, tc.instrument, 10, 0)
    call("handle_track_create", n_cond, 0, 10, 1)   # agent

    # Randomize agent attribute controls so the model is forced to generate notes
    # (leaving these at 0 lets the model predict polyphony=0 → silence)
    agent_id = n_cond
    min_poly = rng.randint(1, 3)
    max_poly = rng.randint(min_poly + 1, min(min_poly + 3, 6))
    min_dur  = rng.randint(1, 4)
    max_dur  = rng.randint(min_dur + 1, min(min_dur + 2, 6))
    density  = rng.randint(1, 8)
    print(f"  agent params: min_poly={min_poly} max_poly={max_poly} "
          f"min_dur={min_dur} max_dur={max_dur} density={density}", flush=True)

    def set_agent_param(name, value):
        path = "/midigpt/track/param/set"
        getattr(orig, "handle_track_param_set")(CLIENT, path, agent_id, name, value)
        getattr(ref,  "handle_track_param_set")(CLIENT, path, agent_id, name, value)

    set_agent_param("min_polyphony_q",    min_poly)
    set_agent_param("max_polyphony_q",    max_poly)
    set_agent_param("min_note_duration_q", min_dur)
    set_agent_param("max_note_duration_q", max_dur)
    set_agent_param("onset_density",      density)

    call("handle_session_start")
    t_start = time.monotonic()

    # Buffer bars — no waiting
    print(f"  Sending {BUFFER_BARS} buffer bars…", flush=True)
    for bar in range(BUFFER_BARS):
        send_bar(bar)

    # Generation-gated bars
    total_bars = BUFFER_BARS + N_BARS_GEN
    bar_iter = range(BUFFER_BARS, total_bars)
    if HAS_TQDM:
        bar_iter = tqdm(bar_iter, desc=f"  {rep_label} gen bars", unit="bar",
                        leave=False, dynamic_ncols=True)

    for bar in bar_iter:
        print(f"\n  → sending bar {bar}  (waiting for both servers…)", flush=True)
        fired_o = orig_cap.wait_for_generated()
        fired_r = ref_cap.wait_for_generated()
        if not fired_o:
            print(f"  [orig] ⚠ timeout waiting for bar {bar-LOOKAHEAD} generation", flush=True)
        if not fired_r:
            print(f"  [ref ] ⚠ timeout waiting for bar {bar-LOOKAHEAD} generation", flush=True)
        send_bar(bar)

    call("handle_session_stop")

    def first_latency(cap):
        ot = cap._open_times
        return (ot[0] - t_start) if ot else float("nan")

    return RepResult(
        rep            = seed,
        n_cond_tracks  = n_cond,
        tracks         = tracks,
        orig_opens     = len(orig_cap.find("/midigpt/generated/open")),
        orig_notes     = len(orig_cap.find("/midigpt/generated/note")),
        orig_errs      = len([c for c in orig_cap.errors() if c == 5]),
        orig_latency   = first_latency(orig_cap),
        orig_intervals = orig_cap.intervals(),
        orig_notes_per_bar = orig_cap.notes_per_bar_list(),
        ref_opens      = len(ref_cap.find("/midigpt/generated/open")),
        ref_notes      = len(ref_cap.find("/midigpt/generated/note")),
        ref_errs       = len([c for c in ref_cap.errors() if c == 5]),
        ref_latency    = first_latency(ref_cap),
        ref_intervals  = ref_cap.intervals(),
        ref_notes_per_bar = ref_cap.notes_per_bar_list(),
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _stat(vals):
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return statistics.mean(v), (statistics.stdev(v) if len(v)>1 else 0.0), min(v), max(v)

def _ms(x):
    return "  —" if (isinstance(x, float) and math.isnan(x)) else f"{x*1000:.0f}ms"

def _f1(x):
    return "  —" if (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def print_results(results):
    W = 100
    print(f"\n{'━'*W}")
    print(f"{'GENERATION COMPARISON  (orig vs refactored)':^{W}}")
    print(f"{'━'*W}")
    print(f"  model_dim={MODEL_DIM}  buffer={BUFFER_BARS}  lookahead={LOOKAHEAD}  "
          f"bars_per_step={BARS_PER_STEP}  gen_bars/session={N_BARS_GEN}  "
          f"gen_timeout={GEN_TIMEOUT}s")
    print(f"{'─'*W}")

    hdr = (f"  {'Rep':>3}  {'Cond':>5}  {'Instruments (first 4)':^22}  "
           f"{'o.bars':>7} {'o.notes':>8} {'o.npb':>6} {'o.sil':>6} {'o.errs':>6}  │  "
           f"{'r.bars':>7} {'r.notes':>8} {'r.npb':>6} {'r.sil':>6} {'r.errs':>6}")
    print(hdr)
    print(f"{'─'*W}")

    for i, r in enumerate(results):
        npb_o = _f1(statistics.mean(r.orig_notes_per_bar)) if r.orig_notes_per_bar else "—"
        npb_r = _f1(statistics.mean(r.ref_notes_per_bar))  if r.ref_notes_per_bar  else "—"
        sil_o = f"{r.orig_silence_rate()*100:.0f}%" if not math.isnan(r.orig_silence_rate()) else "—"
        sil_r = f"{r.ref_silence_rate()*100:.0f}%"  if not math.isnan(r.ref_silence_rate())  else "—"
        insts = ",".join(str(t.instrument) for t in r.tracks[:4])
        if len(r.tracks) > 4: insts += f"+{len(r.tracks)-4}"
        print(
            f"  {i:>3}  {r.n_cond_tracks:>5}  {insts:^22}  "
            f"{r.orig_opens:>7} {r.orig_notes:>8} {npb_o:>6} {sil_o:>6} {r.orig_errs:>6}  │  "
            f"{r.ref_opens:>7} {r.ref_notes:>8} {npb_r:>6} {sil_r:>6} {r.ref_errs:>6}"
        )

    print(f"{'─'*W}")

    orig_npb = [n for r in results for n in r.orig_notes_per_bar]
    ref_npb  = [n for r in results for n in r.ref_notes_per_bar]
    orig_ivs = [v for r in results for v in r.orig_intervals]
    ref_ivs  = [v for r in results for v in r.ref_intervals]
    orig_lat = [r.orig_latency for r in results]
    ref_lat  = [r.ref_latency  for r in results]

    om, osd, omin, omax = _stat(orig_npb)
    rm, rsd, rmin, rmax = _stat(ref_npb)
    olm = _stat(orig_lat)[0]; rlm = _stat(ref_lat)[0]
    oim = _stat(orig_ivs)[0]; rim = _stat(ref_ivs)[0]

    orig_sil_pct = sum(1 for n in orig_npb if n==0)/len(orig_npb)*100 if orig_npb else float("nan")
    ref_sil_pct  = sum(1 for n in ref_npb  if n==0)/len(ref_npb) *100 if ref_npb  else float("nan")

    print(f"\n  {'':32} {'orig':>14} {'refactored':>14}")
    print(f"  {'─'*62}")
    def row(lbl, ov, rv): print(f"  {lbl:<32} {ov:>14} {rv:>14}")
    row("Total bars generated",   str(sum(r.orig_opens for r in results)),  str(sum(r.ref_opens  for r in results)))
    row("Total notes generated",  str(sum(r.orig_notes for r in results)),  str(sum(r.ref_notes  for r in results)))
    row("Total gen errors",       str(sum(r.orig_errs  for r in results)),  str(sum(r.ref_errs   for r in results)))
    print(f"  {'Notes/bar  mean±sd':<32} {om:>7.1f}±{osd:<5.1f}   {rm:>7.1f}±{rsd:<5.1f}")
    print(f"  {'Notes/bar  min…max':<32} {omin:>7.0f}…{omax:<5.0f}   {rmin:>7.0f}…{rmax:<5.0f}")
    print(f"  {'Silent bars':<32} {orig_sil_pct:>13.0f}%  {ref_sil_pct:>13.0f}%")
    row("Mean latency (first bar)", _ms(olm), _ms(rlm))
    row("Mean inter-bar interval",  _ms(oim), _ms(rim))
    if not (math.isnan(oim) or math.isnan(rim)) and rim > 0:
        print(f"  {'Speedup (orig÷ref)':<32} {'':>14} {oim/rim:>13.2f}×")
    print(f"{'━'*W}\n")


def main():
    print(f"=== MIDI-GPT generation check ===")
    print(f"  model_dim={MODEL_DIM}  buffer={BUFFER_BARS}  lookahead={LOOKAHEAD}  "
          f"gen_bars={N_BARS_GEN}  reps={N_REPS}  timeout={GEN_TIMEOUT}s")
    print(f"  ckpt: {CKPT}\n")

    print("Building inference engine (refactored):")
    engine = build_ref_engine()

    base_params = {
        "buffer_bars": BUFFER_BARS, "lookahead_bars": LOOKAHEAD,
        "num_anticipated_bars": BARS_PER_STEP, "model_dim": MODEL_DIM,
        "gen_timeout": GEN_TIMEOUT,
    }
    orig, oc = make_orig(base_params)
    ref,  rc = make_ref(engine, base_params)

    meta_rng = random.Random(0xDEADBEEF)
    results  = []

    for rep in range(N_REPS):
        n_cond  = meta_rng.randint(1, 16)
        tracks  = [random_track_config(meta_rng) for _ in range(n_cond)]
        seed    = meta_rng.randint(0, 2**31)
        label   = f"rep{rep+1}"

        print(f"\n{'─'*60}")
        print(f"Rep {rep+1}/{N_REPS} — {n_cond} cond track(s)")
        for i, t in enumerate(tracks):
            print(f"  track {i}: inst={t.instrument:>3}  npb={t.notes_per_bar:>2}  "
                  f"pitch=[{t.min_pitch},{t.max_pitch}]  vel=[{t.min_vel},{t.max_vel}]")
        print(f"  seed={seed}")
        print(f"{'─'*60}")

        r = run_session(orig, oc, ref, rc, n_cond, tracks, seed, label)
        results.append(r)

        print(f"\n  Rep {rep+1} summary:")
        print(f"    orig: {r.orig_opens} bars / {r.orig_notes} notes / "
              f"{r.orig_errs} errs  lat={_ms(r.orig_latency)}")
        print(f"    ref:  {r.ref_opens} bars / {r.ref_notes} notes / "
              f"{r.ref_errs} errs  lat={_ms(r.ref_latency)}")

    print_results(results)


if __name__ == "__main__":
    main()
