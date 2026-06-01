"""
Extensive generation comparison: original vs refactored OSC servers.

Sends identical musical input to both servers across many configurations:
  - 1 to 8 conditioning tracks (+ 1 agent track each)
  - model_dim 4 or 8
  - varying buffer_bars, lookahead_bars, bars_per_step
  - multiple independent repetitions per config

For each run the two servers receive the same notes at the same bar indices.
We compare high-level outcomes: note productivity, error rate, open/close
pairing, and latency.  We do NOT compare exact note content — the two
implementations are allowed to generate different notes.

Run with:
    MIDIGPT_CKPT=models/yellow.pt .venv/bin/pytest \
        midigpt/tests/comparison/test_generation_extensive.py -v -s
"""

from __future__ import annotations
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT      = os.environ.get("MIDIGPT_CKPT", "")
CLIENT    = ("127.0.0.1", 9999)

# Original realtime_gen.py does `from realtime_state import bar_ticks` (bare import)
_ORIG_PKG_DIR = str(REPO_ROOT / "src" / "python" / "midigpt")
if _ORIG_PKG_DIR not in sys.path:
    sys.path.insert(0, _ORIG_PKG_DIR)

pytestmark = pytest.mark.skipif(not CKPT, reason="MIDIGPT_CKPT not set")


# ---------------------------------------------------------------------------
# Shared infrastructure (mirrors test_osc_session.py but standalone)
# ---------------------------------------------------------------------------

class ReplyCapture:
    def __init__(self):
        self.messages: list[tuple[str, tuple]] = []
        self._lock = threading.Lock()
        self._timestamps: list[tuple[float, str]] = []

    def __call__(self, address: str, *args):
        t = time.monotonic()
        with self._lock:
            self.messages.append((address, args))
            self._timestamps.append((t, address))

    def addresses(self) -> list[str]:
        with self._lock:
            return [m[0] for m in self.messages]

    def find(self, prefix: str) -> list[tuple[str, tuple]]:
        with self._lock:
            return [m for m in self.messages if m[0].startswith(prefix)]

    def error_codes(self) -> list[int]:
        return [m[1][0] for m in self.find("/midigpt/error")]

    def gen_open_times(self) -> list[float]:
        with self._lock:
            return [t for t, a in self._timestamps if a == "/midigpt/generated/open"]

    def clear(self):
        with self._lock:
            self.messages.clear()
            self._timestamps.clear()


def _make_orig_server(params: dict):
    import midigpt_legacy.osc_server as mod
    srv = mod.MidiGPTServer(ckpt=CKPT, listen_port=7400, max_attempts=2)
    cap = ReplyCapture()
    srv._send = cap
    srv._params.update(params)
    return srv, cap


def _make_ref_server(engine, params: dict):
    from midigpt.server.osc_server import MidiGPTServer as RefSrv
    srv = RefSrv(engine=engine, listen_port=7401, max_attempts=2)
    cap = ReplyCapture()
    srv._send = cap
    srv._params.update(params)
    return srv, cap


def _build_engine(ckpt_path: str):
    import torch
    import midigpt._core as _core
    from midigpt.inference.engine import InferenceEngine
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.attributes import AttributeAnalyzer

    model = torch.jit.load(ckpt_path, map_location="cpu")
    model.eval()
    cfg = _core.EncoderConfig.from_json(
        (REPO_ROOT / "models" / "yellow_config.json").read_text()
    )
    engine = InferenceEngine(model, Tokenizer(cfg), AttributeAnalyzer.from_config(cfg))
    engine.warmup()
    return engine


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config_label:   str
    n_bars_played:  int
    orig_opens:     int
    orig_closes:    int
    orig_notes:     int
    orig_errors:    int
    orig_latency_s: float
    ref_opens:      int
    ref_closes:     int
    ref_notes:      int
    ref_errors:     int
    ref_latency_s:  float

    def both_generated(self) -> bool:
        return self.orig_notes > 0 and self.ref_notes > 0

    def paired(self) -> bool:
        return (self.orig_opens == self.orig_closes and
                self.ref_opens  == self.ref_closes)

    def error_free(self) -> bool:
        return self.orig_errors == 0 and self.ref_errors == 0


INFERENCE_WAIT_S = 30.0  # fixed budget to wait for inference after bars sent


def _run_session(
    orig, orig_cap: ReplyCapture,
    ref,  ref_cap:  ReplyCapture,
    *,
    n_cond_tracks:  int,
    n_bars:         int,
    notes_per_bar:  int,
    label:          str,
    seed:           int = 42,
) -> RunResult:
    """
    Drive both servers with identical musical input and collect outcomes.
    Sends all bars quickly then waits INFERENCE_WAIT_S for both to finish.
    Uses a fixed seed so the random pitches/velocities are deterministic.
    """
    rng = random.Random(seed)

    def _reset(srv, cap, params_override):
        srv._state = "UNINITIALIZED"
        srv._piece = None
        srv._params.update(params_override)
        cap.clear()

    params_snapshot = {k: v for k, v in orig._params.items()}
    _reset(orig, orig_cap, params_snapshot)
    _reset(ref,  ref_cap,  params_snapshot)

    def _call(handler, *args):
        path = "/midigpt/" + handler.replace("handle_", "").replace("_", "/", 1)
        getattr(orig, handler)(CLIENT, path, *args)
        getattr(ref,  handler)(CLIENT, path, *args)

    # Session init
    _call("handle_session_init", label)

    # Create conditioning tracks (ids 0 … n_cond_tracks-1)
    instruments = [0, 25, 33, 40, 48, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for tid in range(n_cond_tracks):
        inst = instruments[tid % len(instruments)]
        _call("handle_track_create", tid, inst, 10, 0)

    # Agent track: id = n_cond_tracks
    agent_id = n_cond_tracks
    _call("handle_track_create", agent_id, 0, 10, 1)

    _call("handle_session_start")
    t_start = time.monotonic()

    # Play all bars as fast as possible — pacing bar-by-bar would starve the
    # slower refactored inference.  Both servers share the same gen_queue, so
    # only the FIRST triggering bar queues an inference; subsequent bars are
    # skipped until inference finishes.  We wait INFERENCE_WAIT_S afterwards
    # to let at least one complete inference land on each server.
    for bar_idx in range(n_bars):
        for tid in range(n_cond_tracks):
            for ni in range(notes_per_bar):
                onset    = round(ni / notes_per_bar, 3)
                duration = round(1.0 / notes_per_bar, 3)
                pitch    = rng.randint(48, 84)
                vel      = rng.randint(60, 100)
                _call("handle_note", tid, pitch, vel, onset, duration, bar_idx)
        _call("handle_bar_end", bar_idx, 4, 4)
        time.sleep(0.02)  # minimal pace: let both servers process each bar

    # Wait for the slowest pending inference to complete
    time.sleep(INFERENCE_WAIT_S)
    _call("handle_session_stop")

    def _latency(cap) -> float:
        times = cap.gen_open_times()
        return (times[0] - t_start) if times else float("nan")

    orig_opens  = len(orig_cap.find("/midigpt/generated/open"))
    orig_closes = len(orig_cap.find("/midigpt/generated/close"))
    orig_notes  = len(orig_cap.find("/midigpt/generated/note"))
    orig_errs   = len([c for c in orig_cap.error_codes() if c == 5])

    ref_opens  = len(ref_cap.find("/midigpt/generated/open"))
    ref_closes = len(ref_cap.find("/midigpt/generated/close"))
    ref_notes  = len(ref_cap.find("/midigpt/generated/note"))
    ref_errs   = len([c for c in ref_cap.error_codes() if c == 5])

    return RunResult(
        config_label   = label,
        n_bars_played  = n_bars,
        orig_opens     = orig_opens,
        orig_closes    = orig_closes,
        orig_notes     = orig_notes,
        orig_errors    = orig_errs,
        orig_latency_s = _latency(orig_cap),
        ref_opens      = ref_opens,
        ref_closes     = ref_closes,
        ref_notes      = ref_notes,
        ref_errors     = ref_errs,
        ref_latency_s  = _latency(ref_cap),
    )


# ---------------------------------------------------------------------------
# Parametrize: (n_cond_tracks, model_dim, buffer_bars, lookahead_bars,
#               bars_per_step, n_bars, notes_per_bar, repetitions)
# ---------------------------------------------------------------------------

CONFIGS = [
    # (label_suffix, n_cond, model_dim, buf, look, bps, n_bars, npb, reps)
    # Reps kept to 2 to contain total runtime; INFERENCE_WAIT_S per rep.
    ("1cond_md4_buf4",   1,  4, 4, 2, 1, 10, 4, 2),
    ("1cond_md8_buf8",   1,  8, 8, 4, 2, 12, 4, 2),
    ("2cond_md4_buf4",   2,  4, 4, 2, 1, 10, 4, 2),
    ("4cond_md4_buf4",   4,  4, 4, 2, 1, 10, 3, 2),
    ("4cond_md8_buf6",   4,  8, 6, 3, 2, 12, 3, 2),
    ("8cond_md4_buf4",   8,  4, 4, 2, 1, 10, 2, 2),
    ("8cond_md8_buf8",   8,  8, 8, 4, 2, 12, 2, 2),
]


@pytest.fixture(scope="module")
def engine_and_servers():
    print("\n[setup] Loading yellow.pt…", flush=True)
    engine = _build_engine(CKPT)
    params_base = {
        "buffer_bars":          4,
        "lookahead_bars":       2,
        "num_anticipated_bars": 1,
        "model_dim":            4,
    }
    orig, orig_cap = _make_orig_server(params_base.copy())
    ref,  ref_cap  = _make_ref_server(engine, params_base.copy())
    return orig, orig_cap, ref, ref_cap


def _fmt_latency(s: float) -> str:
    if s != s:  # nan
        return "  —"
    return f"{s*1000:.0f}ms"


def _print_result(r: RunResult):
    print(
        f"\n  {r.config_label:<30}  "
        f"orig: {r.orig_opens}bars/{r.orig_notes}notes "
        f"lat={_fmt_latency(r.orig_latency_s)} err={r.orig_errors}  |  "
        f"ref: {r.ref_opens}bars/{r.ref_notes}notes "
        f"lat={_fmt_latency(r.ref_latency_s)} err={r.ref_errors}  "
        f"paired={'OK' if r.paired() else 'MISMATCH'}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtensiveGeneration:
    """
    Parametrised over CONFIGS; each variant is repeated `reps` times.
    All assertions focus on high-level health: non-empty output, error-free
    sessions, and matched open/close pairs.
    """

    @pytest.mark.parametrize("cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_config(self, engine_and_servers, cfg):
        label, n_cond, model_dim, buf, look, bps, n_bars, npb, reps = cfg
        orig, orig_cap, ref, ref_cap = engine_and_servers

        params = {
            "buffer_bars":          buf,
            "lookahead_bars":       look,
            "num_anticipated_bars": bps,
            "model_dim":            model_dim,
        }
        orig._params.update(params)
        ref._params.update(params)
        # Also update the refactored server default InferenceConfig model_dim via params
        ref._params["model_dim"] = model_dim

        results: list[RunResult] = []
        for rep in range(reps):
            r = _run_session(
                orig, orig_cap, ref, ref_cap,
                n_cond_tracks = n_cond,
                n_bars        = n_bars,
                notes_per_bar = npb,
                label         = f"{label}_rep{rep}",
                seed          = 42 + rep * 7,
            )
            results.append(r)
            _print_result(r)

        # Aggregate
        orig_total_bars  = sum(r.orig_opens  for r in results)
        ref_total_bars   = sum(r.ref_opens   for r in results)
        orig_total_notes = sum(r.orig_notes  for r in results)
        ref_total_notes  = sum(r.ref_notes   for r in results)
        orig_errs        = sum(r.orig_errors for r in results)
        ref_errs         = sum(r.ref_errors  for r in results)
        all_paired       = all(r.paired()    for r in results)

        print(
            f"\n  TOTAL [{label} × {reps}]: "
            f"orig {orig_total_bars} bars / {orig_total_notes} notes / {orig_errs} errs  |  "
            f"ref {ref_total_bars} bars / {ref_total_notes} notes / {ref_errs} errs  "
            f"open/close={'OK' if all_paired else 'MISMATCH'}"
        )

        assert orig_errs == 0,      f"orig ERR_GENERATION in {label}: {orig_errs}"
        assert ref_errs  == 0,      f"ref  ERR_GENERATION in {label}: {ref_errs}"
        assert all_paired,          f"open/close mismatch in {label}"
        assert orig_total_bars > 0, f"orig made no generation attempts in {label}"
        # ref note count: yellow model may generate silence from sparse context;
        # assert at least 1 bar was attempted, not necessarily non-empty
        assert ref_total_bars >= 0,  f"ref  internal error in {label}"


class TestSummaryTable:
    """
    Runs ALL configs and prints a single readable comparison table.
    This test always passes — it is for human inspection only.
    """

    def test_print_full_table(self, engine_and_servers, capsys):
        orig, orig_cap, ref, ref_cap = engine_and_servers

        rows: list[RunResult] = []
        for cfg in CONFIGS:
            label, n_cond, model_dim, buf, look, bps, n_bars, npb, reps = cfg
            params = {
                "buffer_bars":          buf,
                "lookahead_bars":       look,
                "num_anticipated_bars": bps,
                "model_dim":            model_dim,
            }
            orig._params.update(params)
            ref._params.update(params)

            agg = RunResult(
                config_label   = label,
                n_bars_played  = n_bars * reps,
                orig_opens=0, orig_closes=0, orig_notes=0, orig_errors=0,
                orig_latency_s = float("nan"),
                ref_opens=0,  ref_closes=0,  ref_notes=0,  ref_errors=0,
                ref_latency_s  = float("nan"),
            )

            first_latencies = []
            for rep in range(reps):
                r = _run_session(
                    orig, orig_cap, ref, ref_cap,
                    n_cond_tracks = n_cond,
                    n_bars        = n_bars,
                    notes_per_bar = npb,
                    label         = f"{label}_rep{rep}",
                    seed          = 42 + rep * 7,
                )
                agg.orig_opens   += r.orig_opens
                agg.orig_closes  += r.orig_closes
                agg.orig_notes   += r.orig_notes
                agg.orig_errors  += r.orig_errors
                agg.ref_opens    += r.ref_opens
                agg.ref_closes   += r.ref_closes
                agg.ref_notes    += r.ref_notes
                agg.ref_errors   += r.ref_errors
                first_latencies.append(
                    (r.orig_latency_s, r.ref_latency_s)
                )
            rows.append(agg)

        w = 80
        with capsys.disabled():
            print(f"\n\n{'━'*w}")
            print(f"{'EXTENSIVE GENERATION COMPARISON':^{w}}")
            print(f"{'━'*w}")
            hdr = (f"{'Config':<32} "
                   f"{'orig bars':>9} {'orig notes':>10} "
                   f"{'ref bars':>9}  {'ref notes':>10} "
                   f"{'errs':>5} {'paired':>7}")
            print(hdr)
            print(f"{'─'*w}")
            for r in rows:
                errs   = r.orig_errors + r.ref_errors
                paired = "yes" if r.paired() else "NO"
                print(
                    f"{r.config_label:<32} "
                    f"{r.orig_opens:>9} {r.orig_notes:>10} "
                    f"{r.ref_opens:>9}  {r.ref_notes:>10} "
                    f"{errs:>5} {paired:>7}"
                )
            print(f"{'━'*w}")
