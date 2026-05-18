"""
OSC session comparison test.

Sends the same OSC message sequences to both the original midigpt server and
the refactored midigpt_refactor server and compares:
  - State machine transitions (same reply sequences)
  - Error codes for invalid operations
  - Note accumulation / bar finalisation
  - (With model) generated note counts per bar

Both servers are driven by calling their handler methods directly — no actual
UDP sockets — for determinism and speed.  The comparison covers everything
except model inference (which intentionally differs between the two; see
CLAUDE.md architecture notes).

buffer_bars=64 prevents generation from triggering in most tests.
Tests marked skipif require MIDIGPT_CKPT env var to be set.
"""

from __future__ import annotations
import os
import sys
import threading
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = os.environ.get("MIDIGPT_CKPT", "")
CLIENT = ("127.0.0.1", 9999)   # dummy source address for all handler calls

# The original osc_server's realtime_gen.py does `from realtime_state import bar_ticks`
# — a bare import that only works when the midigpt package dir is on sys.path.
_ORIG_PKG_DIR = str(REPO_ROOT / "src" / "python" / "midigpt")
if _ORIG_PKG_DIR not in sys.path:
    sys.path.insert(0, _ORIG_PKG_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ReplyCapture:
    """Replaces MidiGPTServer._send to record all outgoing messages."""

    def __init__(self):
        self.messages: list[tuple[str, tuple]] = []
        self._lock = threading.Lock()

    def __call__(self, address: str, *args):
        with self._lock:
            self.messages.append((address, args))

    def addresses(self) -> list[str]:
        with self._lock:
            return [m[0] for m in self.messages]

    def find(self, prefix: str) -> list[tuple[str, tuple]]:
        with self._lock:
            return [m for m in self.messages if m[0].startswith(prefix)]

    def error_codes(self) -> list[int]:
        return [m[1][0] for m in self.find("/midigpt/error")]

    def clear(self):
        with self._lock:
            self.messages.clear()


def _make_orig_server(params_override: dict | None = None):
    import midigpt.osc_server as orig_mod
    srv = orig_mod.MidiGPTServer(ckpt=CKPT or "no_model", listen_port=7400, max_attempts=1)
    cap = ReplyCapture()
    srv._send = cap
    if params_override:
        srv._params.update(params_override)
    return srv, cap


def _make_ref_server(params_override: dict | None = None):
    from midigpt_refactor.server.osc_server import MidiGPTServer as RefServer

    class _DummyEngine:
        pass

    srv = RefServer(engine=_DummyEngine(), listen_port=7401, max_attempts=1)
    cap = ReplyCapture()
    srv._send = cap
    if params_override:
        srv._params.update(params_override)
    return srv, cap


class BothServers:
    """Drives both servers identically and compares their responses."""

    def __init__(self, orig, orig_cap: ReplyCapture, ref, ref_cap: ReplyCapture):
        self.orig     = orig       # original MidiGPTServer
        self.orig_cap = orig_cap
        self.ref      = ref        # refactored MidiGPTServer
        self.ref_cap  = ref_cap

    def call(self, handler_name: str, *args):
        """Call the named handler on both servers."""
        path = "/midigpt/" + handler_name.replace("handle_", "").replace("_", "/", 1)
        getattr(self.orig, handler_name)(CLIENT, path, *args)
        getattr(self.ref,  handler_name)(CLIENT, path, *args)

    def orig_addresses(self) -> list[str]:
        return self.orig_cap.addresses()

    def ref_addresses(self) -> list[str]:
        return self.ref_cap.addresses()

    def clear(self):
        self.orig_cap.clear()
        self.ref_cap.clear()


@pytest.fixture(scope="module")
def servers():
    orig, orig_cap = _make_orig_server({"buffer_bars": 64})
    ref,  ref_cap  = _make_ref_server( {"buffer_bars": 64})
    return BothServers(orig, orig_cap, ref, ref_cap)


@pytest.fixture
def fresh(servers):
    """Reset both servers to UNINITIALIZED before each test."""
    servers.orig._state = "UNINITIALIZED"
    servers.ref._state  = "UNINITIALIZED"
    servers.orig._piece = None
    servers.ref._piece  = None
    servers.orig._params.update({"buffer_bars": 64, "lookahead_bars": 2,
                                  "num_anticipated_bars": 1})
    servers.ref._params.update( {"buffer_bars": 64, "lookahead_bars": 2,
                                  "num_anticipated_bars": 1})
    servers.clear()
    return servers


# ── Session-flow helpers ───────────────────────────────────────────────────

def _init_session(srv: BothServers):
    srv.call("handle_session_init", "test_session")

def _create_tracks(srv: BothServers):
    srv.call("handle_track_create", 0, 0, 10, 0)    # cond: acoustic grand piano, melodic
    srv.call("handle_track_create", 1, 32, 10, 1)   # agent: acoustic bass (program 32), melodic

def _start_session(srv: BothServers):
    srv.call("handle_session_start")

def _send_bar(srv: BothServers, bar_idx: int, n_notes: int = 2):
    for i in range(n_notes):
        srv.call("handle_note", 0, 60 + i, 80,
                 round(i / n_notes, 3), round(1 / n_notes, 3), bar_idx)
    srv.call("handle_bar_end", bar_idx, 4, 4)

def _stop_session(srv: BothServers):
    srv.call("handle_session_stop")


# ---------------------------------------------------------------------------
# 1. Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:

    def test_init_sends_ready(self, fresh):
        _init_session(fresh)
        assert "/midigpt/session/ready" in fresh.orig_addresses(), "orig: no /ready"
        assert "/midigpt/session/ready" in fresh.ref_addresses(),  "ref:  no /ready"

    def test_start_sends_started(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        fresh.clear()
        _start_session(fresh)
        assert "/midigpt/session/started" in fresh.orig_addresses(), "orig: no /started"
        assert "/midigpt/session/started" in fresh.ref_addresses(),  "ref:  no /started"
        assert fresh.orig._state == "RUNNING", f"orig state: {fresh.orig._state}"
        assert fresh.ref._state  == "RUNNING", f"ref  state: {fresh.ref._state}"

    def test_stop_sends_stopped(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        _stop_session(fresh)
        assert "/midigpt/session/stopped" in fresh.orig_addresses()
        assert "/midigpt/session/stopped" in fresh.ref_addresses()

    def test_reinit_resets_state(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        _init_session(fresh)
        assert "/midigpt/session/ready" in fresh.orig_addresses()
        assert "/midigpt/session/ready" in fresh.ref_addresses()
        assert fresh.orig._state == "INITIALIZING"
        assert fresh.ref._state  == "INITIALIZING"


# ---------------------------------------------------------------------------
# 2. Error codes
# ---------------------------------------------------------------------------

class TestErrorCodes:
    """Same invalid operation → same error code on both servers."""

    def test_start_without_agent_returns_no_agent(self, fresh):
        _init_session(fresh)
        fresh.call("handle_track_create", 0, 0, 10, 0)   # only cond, no agent
        fresh.clear()
        _start_session(fresh)
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes, "orig: expected ERR_NO_AGENT, got none"
        assert ref_codes,  "ref:  expected ERR_NO_AGENT, got none"
        assert orig_codes == ref_codes, f"error codes differ: orig={orig_codes} ref={ref_codes}"
        assert orig_codes[0] == 6, f"expected ERR_NO_AGENT(6), got {orig_codes[0]}"

    def test_duplicate_track_error(self, fresh):
        _init_session(fresh)
        fresh.call("handle_track_create", 0, 0, 10, 0)
        fresh.clear()
        fresh.call("handle_track_create", 0, 0, 10, 0)   # duplicate id=0
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes
        assert orig_codes == ref_codes, f"error codes differ: orig={orig_codes} ref={ref_codes}"
        assert orig_codes[0] == 3, f"expected ERR_DUPLICATE_TRACK(3), got {orig_codes[0]}"

    def test_note_in_initializing_state_is_rejected(self, fresh):
        _init_session(fresh)
        fresh.clear()
        fresh.call("handle_note", 0, 60, 80, 0.0, 0.5, 0)
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes == ref_codes, (
            f"error codes differ for pre-start note: orig={orig_codes} ref={ref_codes}"
        )

    def test_note_to_agent_track_returns_error_8(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        fresh.call("handle_note", 1, 60, 80, 0.0, 0.5, 0)   # track 1 = agent
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes
        assert orig_codes == ref_codes, f"agent note error codes differ: {orig_codes} vs {ref_codes}"
        assert orig_codes[0] == 8, f"expected ERR_AGENT_NOTE(8), got {orig_codes[0]}"

    def test_buffer_bars_below_minimum_rejected(self, fresh):
        _init_session(fresh)
        fresh.clear()
        fresh.call("handle_param_set", "buffer_bars", 1)   # minimum is 2
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes
        assert orig_codes == ref_codes, f"param error codes differ: {orig_codes} vs {ref_codes}"
        assert orig_codes[0] == 4, f"expected ERR_INVALID_PARAM(4), got {orig_codes[0]}"

    def test_multiple_agent_tracks_rejected(self, fresh):
        _init_session(fresh)
        fresh.call("handle_track_create", 0, 0, 10, 1)   # first agent OK
        fresh.clear()
        fresh.call("handle_track_create", 1, 0, 10, 1)   # second agent → error 7
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes
        assert orig_codes == ref_codes, f"multi-agent error codes differ: {orig_codes} vs {ref_codes}"
        assert orig_codes[0] == 7, f"expected ERR_MULTI_AGENT(7), got {orig_codes[0]}"

    def test_unknown_param_name_rejected(self, fresh):
        _init_session(fresh)
        fresh.clear()
        fresh.call("handle_param_set", "nonexistent_param", 42)
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes
        assert orig_codes == ref_codes, f"unknown param error codes differ: {orig_codes} vs {ref_codes}"
        assert orig_codes[0] == 4


# ---------------------------------------------------------------------------
# 3. Parameter management
# ---------------------------------------------------------------------------

class TestParameterManagement:

    def test_valid_param_produces_no_error(self, fresh):
        _init_session(fresh)
        fresh.clear()
        fresh.call("handle_param_set", "temperature", 1.5)
        assert not fresh.orig_cap.error_codes(), "orig: unexpected error on valid param"
        assert not fresh.ref_cap.error_codes(),  "ref:  unexpected error on valid param"

    def test_param_reset_produces_no_error(self, fresh):
        _init_session(fresh)
        fresh.call("handle_param_set", "temperature", 1.8)
        fresh.clear()
        fresh.call("handle_param_reset", "temperature")
        assert not fresh.orig_cap.error_codes(), "param reset produced unexpected error"
        assert not fresh.ref_cap.error_codes()

    def test_param_reset_all_produces_no_error(self, fresh):
        _init_session(fresh)
        fresh.call("handle_param_set", "temperature", 1.8)
        fresh.call("handle_param_set", "lookahead_bars", 3)
        fresh.clear()
        fresh.call("handle_param_reset_all")
        assert not fresh.orig_cap.error_codes()
        assert not fresh.ref_cap.error_codes()


# ---------------------------------------------------------------------------
# 4. Note accumulation and bar finalisation
# ---------------------------------------------------------------------------

class TestNoteAndBarHandling:

    def test_notes_accumulated_across_bars(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        for bar_idx in range(3):
            _send_bar(fresh, bar_idx, n_notes=3)
        assert not fresh.orig_cap.error_codes(), f"orig errors: {fresh.orig_cap.error_codes()}"
        assert not fresh.ref_cap.error_codes(),  f"ref  errors: {fresh.ref_cap.error_codes()}"
        assert fresh.orig._piece.bars_completed == 3
        assert fresh.ref._piece.bars_completed  == 3

    def test_note_count_matches_after_bar_end(self, fresh):
        N = 4
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        _send_bar(fresh, 0, n_notes=N)
        orig_events = fresh.orig._piece._tracks[0].bars[0]["events"]
        ref_events  = fresh.ref._piece._tracks[0].bars[0]["events"]
        assert len(orig_events) == N, f"orig: {len(orig_events)} events, expected {N}"
        assert len(ref_events)  == N, f"ref:  {len(ref_events)} events, expected {N}"

    def test_time_signatures_stored_correctly(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.call("handle_bar_end", 0, 3, 4)   # 3/4
        fresh.call("handle_bar_end", 1, 6, 8)   # 6/8
        assert fresh.orig._piece._ts.get(0) == (3, 4)
        assert fresh.ref._piece._ts.get(0)  == (3, 4)
        assert fresh.orig._piece._ts.get(1) == (6, 8)
        assert fresh.ref._piece._ts.get(1)  == (6, 8)

    def test_cond_track_remove_succeeds(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        fresh.clear()
        fresh.call("handle_track_remove", 0)   # remove cond track
        assert not fresh.orig_cap.error_codes(), "orig: unexpected error removing cond track"
        assert not fresh.ref_cap.error_codes(),  "ref:  unexpected error removing cond track"

    def test_agent_track_cannot_be_removed(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        fresh.clear()
        fresh.call("handle_track_remove", 1)   # attempt to remove agent track
        orig_codes = fresh.orig_cap.error_codes()
        ref_codes  = fresh.ref_cap.error_codes()
        assert orig_codes and ref_codes, "expected error removing agent track"
        assert orig_codes == ref_codes, f"error codes differ: {orig_codes} vs {ref_codes}"


# ---------------------------------------------------------------------------
# 5. Long session (no generation — buffer too high)
# ---------------------------------------------------------------------------

class TestLongSession:

    def test_20_bars_no_errors(self, fresh):
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        for bar_idx in range(20):
            _send_bar(fresh, bar_idx, n_notes=2)
        assert not fresh.orig_cap.error_codes(), f"orig errors: {fresh.orig_cap.error_codes()}"
        assert not fresh.ref_cap.error_codes(),  f"ref  errors: {fresh.ref_cap.error_codes()}"
        assert fresh.orig._piece.bars_completed == 20
        assert fresh.ref._piece.bars_completed  == 20
        _stop_session(fresh)
        assert fresh.orig._state == "STOPPED"
        assert fresh.ref._state  == "STOPPED"

    def test_mixed_time_signatures(self, fresh):
        """3/4 and 6/8 bars interleaved — no errors on either server."""
        _init_session(fresh)
        _create_tracks(fresh)
        _start_session(fresh)
        fresh.clear()
        ts_seq = [(4, 4), (4, 4), (3, 4), (3, 4), (6, 8), (6, 8), (4, 4), (4, 4)]
        for bar_idx, (ts_n, ts_d) in enumerate(ts_seq):
            fresh.call("handle_note", 0, 60, 80, 0.0, 0.25, bar_idx)
            fresh.call("handle_bar_end", bar_idx, ts_n, ts_d)
        assert not fresh.orig_cap.error_codes()
        assert not fresh.ref_cap.error_codes()
        # Verify time signatures stored consistently
        for bar_idx, (ts_n, ts_d) in enumerate(ts_seq):
            assert fresh.orig._piece._ts.get(bar_idx) == (ts_n, ts_d), \
                f"orig bar {bar_idx} ts mismatch"
            assert fresh.ref._piece._ts.get(bar_idx)  == (ts_n, ts_d), \
                f"ref  bar {bar_idx} ts mismatch"


# ---------------------------------------------------------------------------
# 6. With-model: generation output comparison + speed
# ---------------------------------------------------------------------------

def _build_ref_engine(ckpt_path: str):
    """Build InferenceEngine directly from a TorchScript .pt + yellow_config.json.

    The refactored checkpoint loader expects a directory; yellow.pt is a raw
    TorchScript bundle with embedded metadata.  We construct the engine manually
    so the speed test can use the same model as the original server.
    """
    import torch
    import midigpt_refactor._core as _core
    from midigpt_refactor.inference.engine import InferenceEngine
    from midigpt_refactor.tokenizer.tokenizer import Tokenizer
    from midigpt_refactor.attributes import AttributeAnalyzer

    model = torch.jit.load(ckpt_path, map_location="cpu")
    model.eval()

    cfg_path = REPO_ROOT / "models" / "yellow_config.json"
    cfg = _core.EncoderConfig.from_json(cfg_path.read_text())
    tokenizer = Tokenizer(cfg)
    analyzer  = AttributeAnalyzer.from_config(cfg)
    engine    = InferenceEngine(model, tokenizer, analyzer)
    engine.warmup()
    return engine


@pytest.mark.skipif(not CKPT, reason="MIDIGPT_CKPT not set — skipping generation test")
class TestGenerationComparison:
    """
    With yellow.pt, run a short session on both servers and compare:
      - Error counts
      - Generated note counts
      - Time-to-first-generated-bar (latency)
      - Total inference time per generation call
    """

    N_BARS      = 8    # bars to play through
    BUF_BARS    = 4   # buffer before first generation triggers
    LOOK_BARS   = 2   # lookahead
    ANT_BARS    = 1   # bars generated per inference call
    BAR_PACE_S  = 5.0 # seconds between bars (gives inference time to complete)

    @pytest.fixture(scope="class")
    def gen_servers(self):
        print("\nLoading yellow.pt for generation tests…", flush=True)
        engine = _build_ref_engine(CKPT)

        params = {
            "buffer_bars":          self.BUF_BARS,
            "lookahead_bars":       self.LOOK_BARS,
            "num_anticipated_bars": self.ANT_BARS,
        }
        orig, orig_cap = _make_orig_server(params)
        ref,  ref_cap  = _make_ref_server(params)
        ref._engine = engine

        return BothServers(orig, orig_cap, ref, ref_cap)

    @pytest.fixture(autouse=True)
    def _reset_gen(self, gen_servers):
        """Fresh session for every test in this class."""
        gen_servers.orig._state = "UNINITIALIZED"
        gen_servers.ref._state  = "UNINITIALIZED"
        gen_servers.orig._piece = None
        gen_servers.ref._piece  = None
        gen_servers.orig._params.update({
            "buffer_bars": self.BUF_BARS,
            "lookahead_bars": self.LOOK_BARS,
            "num_anticipated_bars": self.ANT_BARS,
        })
        gen_servers.ref._params.update({
            "buffer_bars": self.BUF_BARS,
            "lookahead_bars": self.LOOK_BARS,
            "num_anticipated_bars": self.ANT_BARS,
        })
        gen_servers.clear()
        yield

    def _run_session(self, srv: BothServers, n_bars: int):
        """Full session: init → tracks → start → bars → stop.
        Captures timestamps on each /generated/open message.
        """
        import time

        # Patch _send to add timestamps
        orig_times: list[float] = []
        ref_times:  list[float] = []
        _orig_send_real = srv.orig._send
        _ref_send_real  = srv.ref._send

        def _orig_ts(addr, *args):
            if addr == "/midigpt/generated/open":
                orig_times.append(time.monotonic())
            _orig_send_real(addr, *args)

        def _ref_ts(addr, *args):
            if addr == "/midigpt/generated/open":
                ref_times.append(time.monotonic())
            _ref_send_real(addr, *args)

        srv.orig._send = _orig_ts
        srv.ref._send  = _ref_ts

        t0_orig = t0_ref = None

        _init_session(srv)
        _create_tracks(srv)
        _start_session(srv)

        t0_orig = t0_ref = time.monotonic()

        # Scale-like pitches cycling across bars to give the model rich context.
        PITCHES = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 59]
        N_NOTES = 8  # eighth notes

        for bar_idx in range(n_bars):
            # Send 8 notes (eighth-note grid) with scale-like pitches
            for ni in range(N_NOTES):
                onset    = round(ni / N_NOTES, 3)
                duration = round(1.0 / N_NOTES, 3)
                pitch    = PITCHES[(bar_idx * N_NOTES + ni) % len(PITCHES)]
                vel      = 80 if ni % 2 == 0 else 65
                srv.call("handle_note", 0, pitch, vel, onset, duration, bar_idx)
            srv.call("handle_bar_end", bar_idx, 4, 4)
            # Pace bar delivery to give inference time to complete.
            # After the buffer fills, slow down so each generation finishes
            # before the next bar arrives.
            if bar_idx >= self.BUF_BARS - 1:
                time.sleep(self.BAR_PACE_S)
            else:
                time.sleep(0.05)

        time.sleep(self.BAR_PACE_S)   # wait for last pending inference
        _stop_session(srv)

        # Restore real sends
        srv.orig._send = _orig_send_real
        srv.ref._send  = _ref_send_real

        return t0_orig, t0_ref, orig_times, ref_times

    def test_generation_speed_and_counts(self, gen_servers, capsys):
        import time

        srv = gen_servers
        t0, _, orig_times, ref_times = self._run_session(srv, self.N_BARS)

        orig_notes  = srv.orig_cap.find("/midigpt/generated/note")
        ref_notes   = srv.ref_cap.find("/midigpt/generated/note")
        orig_opens  = srv.orig_cap.find("/midigpt/generated/open")
        ref_opens   = srv.ref_cap.find("/midigpt/generated/open")
        orig_closes = srv.orig_cap.find("/midigpt/generated/close")
        ref_closes  = srv.ref_cap.find("/midigpt/generated/close")
        orig_errs   = [c for c in srv.orig_cap.error_codes() if c == 5]
        ref_errs    = [c for c in srv.ref_cap.error_codes()  if c == 5]

        # Latency: time from session start to first generated bar
        orig_latency = (orig_times[0] - t0) if orig_times else float("nan")
        ref_latency  = (ref_times[0]  - t0) if ref_times  else float("nan")

        # Inter-bar intervals: time between consecutive generated bars
        def _intervals(ts):
            return [ts[i+1] - ts[i] for i in range(len(ts)-1)] if len(ts) > 1 else []

        orig_ivs = _intervals(orig_times)
        ref_ivs  = _intervals(ref_times)

        def _ms(s): return f"{s*1000:.0f}ms"
        def _avg(lst): return sum(lst)/len(lst) if lst else float("nan")

        with capsys.disabled():
            print(f"\n{'─'*55}")
            print(f"{'OSC Generation Speed Comparison':^55}")
            print(f"{'─'*55}")
            print(f"{'':30} {'orig':>10} {'refactor':>10}")
            print(f"{'─'*55}")
            print(f"{'Bars generated':30} {len(orig_opens):>10} {len(ref_opens):>10}")
            print(f"{'Notes generated':30} {len(orig_notes):>10} {len(ref_notes):>10}")
            print(f"{'Latency (first bar)':30} {_ms(orig_latency):>10} {_ms(ref_latency):>10}")
            if orig_ivs:
                print(f"{'Avg time between bars':30} {_ms(_avg(orig_ivs)):>10}", end="")
                print(f" {_ms(_avg(ref_ivs)) if ref_ivs else 'N/A':>10}")
            if orig_ivs and ref_ivs:
                speedup = _avg(orig_ivs) / _avg(ref_ivs)
                print(f"{'Speedup (orig/refactor)':30} {'':>10} {speedup:>9.2f}×")
            print(f"{'ERR_GENERATION errors':30} {len(orig_errs):>10} {len(ref_errs):>10}")
            print(f"{'open/close paired':30}", end="")
            print(f" {'yes' if len(orig_opens)==len(orig_closes) else 'NO':>10}", end="")
            print(f" {'yes' if len(ref_opens)==len(ref_closes) else 'NO':>10}")
            print(f"{'─'*55}")

        # Assertions — check health, not exact note counts (model may generate silence)
        assert not orig_errs, f"orig: ERR_GENERATION errors occurred"
        assert not ref_errs,  f"ref:  ERR_GENERATION errors occurred"
        assert len(orig_opens) > 0, "orig: no generation attempts at all"
        assert len(ref_opens)  > 0, "ref:  no generation attempts at all"
        assert len(orig_opens) == len(orig_closes), "orig: open/close mismatch"
        assert len(ref_opens)  == len(ref_closes),  "ref:  open/close mismatch"

    def test_generation_bar_messages_paired(self, gen_servers):
        """Every /generated/open must have a matching /generated/close."""
        self._run_session(gen_servers, self.N_BARS)

        orig_opens  = gen_servers.orig_cap.find("/midigpt/generated/open")
        orig_closes = gen_servers.orig_cap.find("/midigpt/generated/close")
        ref_opens   = gen_servers.ref_cap.find("/midigpt/generated/open")
        ref_closes  = gen_servers.ref_cap.find("/midigpt/generated/close")

        assert len(orig_opens) == len(orig_closes), (
            f"orig: {len(orig_opens)} opens vs {len(orig_closes)} closes"
        )
        assert len(ref_opens) == len(ref_closes), (
            f"ref: {len(ref_opens)} opens vs {len(ref_closes)} closes"
        )


# ---------------------------------------------------------------------------
# Live OSC test — both servers running on real UDP sockets, test talks to them
# over the network and measures actual round-trip generation latency per bar.
# ---------------------------------------------------------------------------

import socket
import time as _time
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_packet import OscPacket


class LiveOSCEndpoint:
    """UDP socket bound to a fixed local port.  Sends OSC to a target server
    and concurrently receives replies on the same port (so the server's
    'reply to source address' lands back here).  Records every reply with a
    monotonic timestamp."""

    def __init__(self, local_port: int, server_addr: tuple[str, int]):
        self.server_addr = server_addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", local_port))
        self.received: list[tuple[float, str, list]] = []
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._listen, daemon=True)
        self._t.start()

    def _listen(self):
        self.sock.settimeout(0.2)
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            t = _time.monotonic()
            try:
                pkt = OscPacket(data)
                for tm in pkt.messages:
                    m = tm.message
                    with self._lock:
                        self.received.append((t, m.address, list(m.params)))
            except Exception:
                pass

    def send(self, address: str, *args):
        b = OscMessageBuilder(address=address)
        for a in args:
            b.add_arg(a)
        self.sock.sendto(b.build().dgram, self.server_addr)

    def find(self, address: str) -> list[tuple[float, list]]:
        with self._lock:
            return [(t, args) for t, addr, args in self.received if addr == address]

    def close(self):
        self._stop = True
        try:
            self.sock.close()
        except Exception:
            pass


@pytest.mark.skipif(not CKPT, reason="MIDIGPT_CKPT not set — skipping live OSC test")
class TestGenerationLiveOSC:
    """Boots BOTH servers on real UDP ports and drives them as a network client.
    Measures wall-clock round-trip latency per bar: time from sending
    /midigpt/bar/end → receiving the matching /midigpt/generated/close."""

    ORIG_SRV_PORT = 17400
    REF_SRV_PORT  = 17401
    ORIG_CLI_PORT = 18400
    REF_CLI_PORT  = 18401

    N_BARS    = 16
    BUF_BARS  = 4
    LOOK_BARS = 2
    ANT_BARS  = 1
    WAIT_TIMEOUT = 30.0  # max seconds to wait for /generated/close per bar

    @pytest.fixture(scope="class")
    def live_servers(self):
        print("\nLoading yellow.pt for live OSC test…", flush=True)
        engine = _build_ref_engine(CKPT)

        import midigpt.osc_server as orig_mod
        from midigpt_refactor.server.osc_server import MidiGPTServer as RefServer

        orig = orig_mod.MidiGPTServer(ckpt=CKPT, listen_port=self.ORIG_SRV_PORT, max_attempts=1)
        ref  = RefServer(engine=engine, listen_port=self.REF_SRV_PORT, max_attempts=1)
        for srv in (orig, ref):
            srv._params.update({
                "buffer_bars":         self.BUF_BARS,
                "lookahead_bars":      self.LOOK_BARS,
                "num_anticipated_bars": self.ANT_BARS,
            })

        orig_t = threading.Thread(target=orig.serve, kwargs={"host": "127.0.0.1"}, daemon=True)
        ref_t  = threading.Thread(target=ref.serve,  kwargs={"host": "127.0.0.1"}, daemon=True)
        orig_t.start(); ref_t.start()
        _time.sleep(0.5)
        yield orig, ref

    # Per-config sweep.  Each entry defines a distinct generation scenario.
    # Both orig and ref receive byte-identical OSC sequences for every config.
    CONFIGS: list[dict] = [
        # (name, human_tracks=[(instrument_gm, track_type), ...], params={})
        {"name": "1-human-piano-default",
         "humans": [(0,  10)], "params": {}},
        {"name": "1-human-piano-temp0.7",
         "humans": [(0,  10)], "params": {"temperature": 0.7}},
        {"name": "1-human-piano-temp1.4",
         "humans": [(0,  10)], "params": {"temperature": 1.4}},
        {"name": "2-human-piano+bass",
         "humans": [(0,  10), (32, 10)], "params": {}},
        {"name": "3-human-piano+bass+strings",
         "humans": [(0,  10), (32, 10), (48, 10)], "params": {}},
    ]

    def _await(self, ep: LiveOSCEndpoint, address: str,
               predicate, since_idx: int, timeout: float) -> tuple[float, list] | None:
        """Block until a message at *address* matching *predicate* arrives after
        index *since_idx*.  Returns (recv_time, args) or None on timeout."""
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            with ep._lock:
                msgs = list(ep.received[since_idx:])
            for t, addr, args in msgs:
                if addr == address and predicate(args):
                    return t, args
            _time.sleep(0.01)
        return None

    def _drive_session(self, server_addr, client_port, config):
        ep = LiveOSCEndpoint(client_port, server_addr)
        try:
            agent_id = len(config["humans"])  # last id = agent
            ep.send("/midigpt/session/init", "live")
            _time.sleep(0.1)
            # Conditioning (human) tracks
            for i, (inst, ttype) in enumerate(config["humans"]):
                ep.send("/midigpt/track/create", i, inst, ttype, 0)
            # Agent track — bass so it differs from human pianos
            ep.send("/midigpt/track/create", agent_id, 32, 10, 1)
            _time.sleep(0.05)
            # Apply per-session params
            for k, v in config["params"].items():
                ep.send("/midigpt/param/set", k, float(v))
            ep.send("/midigpt/session/start")
            _time.sleep(0.1)

            # Scale-like content for each human track, distinct pitch range
            SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 59]
            N_NOTES = 8

            bar_send_t: dict[int, float] = {}
            latencies: list[tuple[int, float]] = []

            for bar_idx in range(self.N_BARS):
                for h_id, (inst, _ttype) in enumerate(config["humans"]):
                    offset = -12 * h_id   # each extra track an octave lower
                    for ni in range(N_NOTES):
                        pitch = SCALE[(bar_idx * N_NOTES + ni) % len(SCALE)] + offset
                        vel   = 80 if ni % 2 == 0 else 65
                        ep.send("/midigpt/note", h_id, pitch, vel,
                                round(ni / N_NOTES, 3),
                                round(1.0 / N_NOTES, 3),
                                bar_idx)
                # Snapshot index BEFORE send so _await won't match earlier closes.
                with ep._lock:
                    pre_idx = len(ep.received)
                bar_send_t[bar_idx] = _time.monotonic()
                ep.send("/midigpt/bar/end", bar_idx, 4, 4)

                # After playhead crosses the buffer, every /bar/end triggers a
                # generation for target_bar = bar_idx + 1 + LOOK_BARS.
                if bar_idx >= self.BUF_BARS - 1:
                    expected_gen_bar = bar_idx + 1 + self.LOOK_BARS
                    res = self._await(
                        ep, "/midigpt/generated/close",
                        predicate=lambda a, gb=expected_gen_bar: int(a[1]) == gb,
                        since_idx=pre_idx,
                        timeout=self.WAIT_TIMEOUT,
                    )
                    if res is not None:
                        recv_t, _ = res
                        latencies.append((expected_gen_bar, recv_t - bar_send_t[bar_idx]))
                else:
                    _time.sleep(0.02)

            ep.send("/midigpt/session/stop")
            _time.sleep(0.2)
            return latencies, ep.find("/midigpt/generated/note")
        finally:
            ep.close()

    def test_live_round_trip_per_config(self, live_servers, capsys):
        def _ms(s): return f"{s*1000:.0f}ms"
        def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
        def _med(lst):
            if not lst: return float("nan")
            s = sorted(lst); n = len(s)
            return s[n//2] if n % 2 else 0.5 * (s[n//2-1] + s[n//2])

        rows: list[dict] = []
        for cfg in self.CONFIGS:
            orig_lat, orig_notes = self._drive_session(
                ("127.0.0.1", self.ORIG_SRV_PORT), self.ORIG_CLI_PORT, cfg
            )
            ref_lat, ref_notes = self._drive_session(
                ("127.0.0.1", self.REF_SRV_PORT), self.REF_CLI_PORT, cfg
            )
            ot = [t for _, t in orig_lat]
            rt = [t for _, t in ref_lat]
            rows.append({
                "name":       cfg["name"],
                "orig_gens":  len(orig_lat),
                "ref_gens":   len(ref_lat),
                "orig_notes": len(orig_notes),
                "ref_notes":  len(ref_notes),
                "orig_first": ot[0] if ot else float("nan"),
                "ref_first":  rt[0] if rt else float("nan"),
                "orig_mean":  _avg(ot[1:]) if len(ot) > 1 else float("nan"),
                "ref_mean":   _avg(rt[1:]) if len(rt) > 1 else float("nan"),
                "orig_med":   _med(ot[1:]) if len(ot) > 1 else float("nan"),
                "ref_med":    _med(rt[1:]) if len(rt) > 1 else float("nan"),
            })

        with capsys.disabled():
            width = 32 + 5 * 14
            print(f"\n{'─'*width}")
            print(f"{'Live OSC Round-Trip Latency  (16 bars, request/response)':^{width}}")
            print(f"{'─'*width}")
            print(f"{'config':32}{'gens':>14}{'notes':>14}{'first':>14}{'mean rest':>14}{'med rest':>14}")
            print(f"{'─'*width}")
            for r in rows:
                print(f"{r['name'] + '  [orig]':32}"
                      f"{r['orig_gens']:>14}"
                      f"{r['orig_notes']:>14}"
                      f"{_ms(r['orig_first']):>14}"
                      f"{_ms(r['orig_mean']):>14}"
                      f"{_ms(r['orig_med']):>14}")
                print(f"{r['name'] + '  [ref ]':32}"
                      f"{r['ref_gens']:>14}"
                      f"{r['ref_notes']:>14}"
                      f"{_ms(r['ref_first']):>14}"
                      f"{_ms(r['ref_mean']):>14}"
                      f"{_ms(r['ref_med']):>14}")
                if r['orig_mean'] == r['orig_mean'] and r['ref_mean'] == r['ref_mean']:
                    sp = r['orig_mean'] / r['ref_mean']
                    print(f"{'  → speedup (orig/ref)':32}{'':>14}{'':>14}{'':>14}{sp:>13.2f}×{'':>14}")
                print(f"{'─'*width}")

        # Sanity: each config must produce at least one generation on each side.
        for r in rows:
            assert r["orig_gens"] > 0, f"orig produced no gens for {r['name']}"
            assert r["ref_gens"]  > 0, f"ref  produced no gens for {r['name']}"

    # ------------------------------------------------------------------
    # Focused scenario: piano (chord progression) + bass (monophonic root
    # motion) over a looped 4-bar I-V-vi-IV in C major, agent generates a
    # guitar (GM 24) track for 16 bars.  Both servers receive identical OSC
    # streams; outputs are written to MIDI files and per-bar latency
    # reported.
    # ------------------------------------------------------------------

    GUITAR_PROGRESSION_BARS = 16
    PIANO_ID = 0
    BASS_ID  = 1
    AGENT_ID = 2

    # I-V-vi-IV in C major, repeated 4 times to fill 16 bars.
    # Each entry is (chord pitches for piano, bass root pitch).
    CHORD_PROGRESSION = [
        ([60, 64, 67],     48),   # C  major / C2
        ([55, 59, 62],     43),   # G  major / G2
        ([57, 60, 64],     45),   # Am       / A2
        ([53, 57, 60],     41),   # F  major / F2
    ]

    def _send_chord_progression_bar(self, ep: LiveOSCEndpoint, bar_idx: int):
        """Send notes for one bar of the looped chord progression to both
        the piano and bass tracks."""
        chord_pitches, bass_root = self.CHORD_PROGRESSION[bar_idx % 4]
        # Piano: 4 quarter-note chord hits per bar
        N_HITS = 4
        for hit in range(N_HITS):
            onset    = round(hit / N_HITS, 4)
            duration = round(1.0 / N_HITS, 4)
            for p in chord_pitches:
                ep.send("/midigpt/note", self.PIANO_ID, p, 75,
                        onset, duration, bar_idx)
        # Bass: monophonic root-fifth-root-octave walking pattern
        BASS_PATTERN = [bass_root, bass_root + 7, bass_root, bass_root + 12]
        for hit, pitch in enumerate(BASS_PATTERN):
            onset    = round(hit / N_HITS, 4)
            duration = round(1.0 / N_HITS, 4)
            ep.send("/midigpt/note", self.BASS_ID, pitch, 85,
                    onset, duration, bar_idx)

    def _drive_progression_session(self, server_addr, client_port):
        """Run the piano+bass+agent-guitar scenario.  Returns
        (latencies, captured_human_notes, captured_agent_notes)."""
        ep = LiveOSCEndpoint(client_port, server_addr)
        try:
            ep.send("/midigpt/session/init", "progression")
            _time.sleep(0.1)
            ep.send("/midigpt/track/create", self.PIANO_ID, 0,  10, 0)  # piano cond
            ep.send("/midigpt/track/create", self.BASS_ID,  32, 10, 0)  # bass  cond
            ep.send("/midigpt/track/create", self.AGENT_ID, 24, 10, 1)  # guitar agent
            _time.sleep(0.05)
            ep.send("/midigpt/session/start")
            _time.sleep(0.1)

            human_notes: list[tuple[int, int, int, float, float, int]] = []
            # (track_id, pitch, velocity, onset, duration, bar_idx)
            bar_send_t: dict[int, float] = {}
            latencies: list[tuple[int, float]] = []

            for bar_idx in range(self.GUITAR_PROGRESSION_BARS):
                # Record what we're sending (so we can write human tracks to MIDI)
                chord_pitches, bass_root = self.CHORD_PROGRESSION[bar_idx % 4]
                for hit in range(4):
                    onset    = round(hit / 4, 4)
                    duration = round(1.0 / 4, 4)
                    for p in chord_pitches:
                        human_notes.append((self.PIANO_ID, p, 75, onset, duration, bar_idx))
                for hit, pitch in enumerate([bass_root, bass_root+7, bass_root, bass_root+12]):
                    onset    = round(hit / 4, 4)
                    duration = round(1.0 / 4, 4)
                    human_notes.append((self.BASS_ID, pitch, 85, onset, duration, bar_idx))

                self._send_chord_progression_bar(ep, bar_idx)
                with ep._lock:
                    pre_idx = len(ep.received)
                bar_send_t[bar_idx] = _time.monotonic()
                ep.send("/midigpt/bar/end", bar_idx, 4, 4)

                if bar_idx >= self.BUF_BARS - 1:
                    expected_gen_bar = bar_idx + 1 + self.LOOK_BARS
                    res = self._await(
                        ep, "/midigpt/generated/close",
                        predicate=lambda a, gb=expected_gen_bar: int(a[1]) == gb,
                        since_idx=pre_idx,
                        timeout=self.WAIT_TIMEOUT,
                    )
                    if res is not None:
                        recv_t, _ = res
                        latencies.append((expected_gen_bar, recv_t - bar_send_t[bar_idx]))
                else:
                    _time.sleep(0.02)

            ep.send("/midigpt/session/stop")
            _time.sleep(0.3)

            agent_notes: list[tuple[int, int, int, float, float]] = []
            # /midigpt/generated/note args: track_id, bar_index, pitch, vel, onset, dur
            for _, args in ep.find("/midigpt/generated/note"):
                if len(args) >= 6:
                    agent_notes.append((
                        int(args[1]), int(args[2]), int(args[3]),
                        float(args[4]), float(args[5]),
                    ))
            return latencies, human_notes, agent_notes
        finally:
            ep.close()

    def _write_session_midi(self, path: str,
                            human_notes: list,
                            agent_notes: list):
        """Write the human + agent material to a MIDI file using mido."""
        import mido
        TPB  = 480
        BPB  = 4          # 4/4
        BAR_TICKS = TPB * BPB
        mid  = mido.MidiFile(ticks_per_beat=TPB)

        def _build_track(name: str, program: int, channel: int,
                         note_list: list[tuple[int, int, float, float, int]]):
            """note_list entries: (pitch, vel, onset_frac, dur_frac, bar_idx)."""
            tr = mido.MidiTrack()
            tr.append(mido.MetaMessage("track_name", name=name, time=0))
            tr.append(mido.Message("program_change", channel=channel,
                                   program=program, time=0))
            # Build absolute-time events
            events: list[tuple[int, str, int, int]] = []
            for pitch, vel, onset, dur, bar in note_list:
                onset_tick = int(round(bar * BAR_TICKS + onset * BAR_TICKS))
                off_tick   = int(round(onset_tick + dur * BAR_TICKS))
                events.append((onset_tick, "on",  pitch, vel))
                events.append((off_tick,   "off", pitch, 0))
            events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
            prev = 0
            for tick, kind, pitch, vel in events:
                delta = max(0, tick - prev)
                msg_type = "note_on" if kind == "on" else "note_off"
                tr.append(mido.Message(msg_type, channel=channel, note=pitch,
                                       velocity=vel, time=delta))
                prev = tick
            return tr

        piano = [(p, v, o, d, b) for (tid, p, v, o, d, b) in human_notes
                 if tid == self.PIANO_ID]
        bass  = [(p, v, o, d, b) for (tid, p, v, o, d, b) in human_notes
                 if tid == self.BASS_ID]
        guitar = [(p, v, o, d, b) for (b, p, v, o, d) in agent_notes]

        mid.tracks.append(_build_track("piano",  0,  0, piano))
        mid.tracks.append(_build_track("bass",   32, 1, bass))
        mid.tracks.append(_build_track("guitar", 24, 2, guitar))
        mid.save(path)

    def test_progression_with_guitar_agent(self, live_servers, capsys):
        out_dir = REPO_ROOT / "tmp_live_osc_out"
        out_dir.mkdir(exist_ok=True)

        orig_lat, orig_human, orig_agent = self._drive_progression_session(
            ("127.0.0.1", self.ORIG_SRV_PORT), self.ORIG_CLI_PORT
        )
        ref_lat, ref_human, ref_agent = self._drive_progression_session(
            ("127.0.0.1", self.REF_SRV_PORT), self.REF_CLI_PORT
        )

        orig_path = out_dir / "progression_orig.mid"
        ref_path  = out_dir / "progression_ref.mid"
        self._write_session_midi(str(orig_path), orig_human, orig_agent)
        self._write_session_midi(str(ref_path),  ref_human,  ref_agent)

        def _ms(s): return f"{s*1000:.0f}ms"
        def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
        def _med(lst):
            if not lst: return float("nan")
            s = sorted(lst); n = len(s)
            return s[n//2] if n % 2 else 0.5 * (s[n//2-1] + s[n//2])

        ot = [t for _, t in orig_lat]
        rt = [t for _, t in ref_lat]

        with capsys.disabled():
            width = 70
            print(f"\n{'─'*width}")
            print(f"{'Piano (chords) + Bass (mono) → Agent generates guitar  (16 bars)':^{width}}")
            print(f"{'─'*width}")
            print(f"{'':30} {'orig':>18} {'refactor':>18}")
            print(f"{'─'*width}")
            print(f"{'Generations completed':30} {len(orig_lat):>18} {len(ref_lat):>18}")
            print(f"{'Guitar notes generated':30} {len(orig_agent):>18} {len(ref_agent):>18}")
            print(f"{'First-bar latency':30} "
                  f"{_ms(ot[0]) if ot else 'N/A':>18} {_ms(rt[0]) if rt else 'N/A':>18}")
            print(f"{'Subsequent-bar mean':30} "
                  f"{_ms(_avg(ot[1:])) if len(ot)>1 else 'N/A':>18} "
                  f"{_ms(_avg(rt[1:])) if len(rt)>1 else 'N/A':>18}")
            print(f"{'Subsequent-bar median':30} "
                  f"{_ms(_med(ot[1:])) if len(ot)>1 else 'N/A':>18} "
                  f"{_ms(_med(rt[1:])) if len(rt)>1 else 'N/A':>18}")
            if len(ot) > 1 and len(rt) > 1 and _avg(rt[1:]) > 0:
                sp = _avg(ot[1:]) / _avg(rt[1:])
                print(f"{'Speedup (orig/refactor)':30} {'':>18} {sp:>17.2f}×")
            print(f"{'─'*width}")
            print(f"MIDI written:")
            print(f"  orig → {orig_path}")
            print(f"  ref  → {ref_path}")
            print(f"{'─'*width}")

        assert orig_lat, "orig: no generations completed"
        assert ref_lat,  "ref:  no generations completed"
