"""
python_scripts_for_testing/simulate_osc_session.py

End-to-end OSC simulation: drives the MIDI-GPT OSC server with pre-composed
MIDI as if it were a live performance, then reports full system diagnostics.

Usage examples
--------------
# Live mode — notes trickle in at realistic onset times (1x real-time):
python simulate_osc_session.py --midi tests/short_midi/Funkytown.mid \\
    --ckpt /path/to/model.pt --mode live

# Batched mode — all notes for a bar arrive at bar-start, 4x real-time:
python simulate_osc_session.py --midi tests/short_midi/Funkytown.mid \\
    --ckpt /path/to/model.pt --mode batched --realtime_factor 4.0

# Stress test — 10x speed, tight lookahead:
python simulate_osc_session.py --midi tests/short_midi/Funkytown.mid \\
    --ckpt /path/to/model.pt --mode live --realtime_factor 10.0 \\
    --lookahead 1 --model_dim 4

# Save full event log:
python simulate_osc_session.py ... --log_json /tmp/osc_diag.json

Modes
-----
live     Notes are sent at their actual intra-bar onset positions.  Models
         a performer who plays the notes as they land.  Bar-end is sent at
         the true end of the bar.

batched  All notes for a bar are sent at bar-start in one burst, then the
         client sleeps until bar-end.  Models a quantised MIDI sequencer or
         a client that batches notes per bar before forwarding them.

Diagnostics reported
--------------------
Per-message timing, per-bar summary, per-generation latency.
Key question answered: given bar_duration / realtime_factor seconds per bar,
can the model generate fast enough to stay lookahead bars ahead?

  Slack (ms) = deadline - latency
  deadline   = lookahead * bar_duration_ms
  Positive slack = on-time.  Negative slack = late (model can't keep up).
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "python_scripts"))
sys.path.insert(0, str(_REPO / "build"))
sys.path.insert(0, str(_REPO / "build_symusic"))

try:
    import midigpt  # type: ignore[import]
    HAS_MIDIGPT = True
except ImportError:
    HAS_MIDIGPT = False

try:
    from pythonosc.osc_message_builder import OscMessageBuilder
    from pythonosc.osc_message import OscMessage
    HAS_PYTHONOSC = True
except ImportError:
    HAS_PYTHONOSC = False

from osc_server import MidiGPTServer

# ── Timing helpers ────────────────────────────────────────────────────────────

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def bar_duration_ms(ts_num: int, ts_den: int, bpm: float) -> float:
    """Duration of one bar in milliseconds."""
    quarter_notes_per_bar = ts_num * 4.0 / ts_den
    return quarter_notes_per_bar * 60_000.0 / bpm


# ── Diagnostics data model ────────────────────────────────────────────────────

@dataclass
class Event:
    """One logged event."""
    kind:  str
    t_abs: float   # perf_counter() at event time (ms)
    t_rel: float   # ms since session_start
    data:  dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "t_abs": self.t_abs,
                "t_rel": self.t_rel, **self.data}


@dataclass
class BarRecord:
    bar_index: int
    ts_num:    int   = 4
    ts_den:    int   = 4
    bpm:       float = 120.0
    start_t:   float = 0.0   # ms (abs) when bar started
    end_t:     float = 0.0   # ms (abs) when bar_end was sent
    notes_sent: int  = 0

    # Generation targeting this bar as the output bar
    gen_trigger_t: Optional[float] = None  # ms when generation fired
    gen_open_t:    Optional[float] = None  # ms when /generated/open arrived
    gen_close_t:   Optional[float] = None  # ms when /generated/close arrived
    gen_notes:     int = 0
    gen_dropped:   bool = False

    # Set after bar_duration is known
    _deadline_ms: Optional[float] = None

    @property
    def bar_dur_ms(self) -> float:
        return bar_duration_ms(self.ts_num, self.ts_den, self.bpm)

    @property
    def gen_latency_ms(self) -> Optional[float]:
        """trigger → /generated/open (first byte of result)."""
        if self.gen_trigger_t is not None and self.gen_open_t is not None:
            return self.gen_open_t - self.gen_trigger_t
        return None

    @property
    def gen_total_ms(self) -> Optional[float]:
        """trigger → /generated/close (all notes delivered)."""
        if self.gen_trigger_t is not None and self.gen_close_t is not None:
            return self.gen_close_t - self.gen_trigger_t
        return None

    @property
    def deadline_ms(self) -> Optional[float]:
        return self._deadline_ms

    @property
    def slack_ms(self) -> Optional[float]:
        if self.gen_total_ms is not None and self._deadline_ms is not None:
            return self._deadline_ms - self.gen_total_ms
        return None

    @property
    def on_time(self) -> Optional[bool]:
        s = self.slack_ms
        return None if s is None else (s >= 0)


class DiagnosticsLog:
    """Thread-safe event recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Event] = []
        self._bars:   Dict[int, BarRecord] = {}
        self._session_start_t: float = 0.0
        self._errors: List[dict] = []

    def session_start(self) -> None:
        self._session_start_t = _now_ms()

    def record(self, kind: str, **data) -> float:
        t = _now_ms()
        rel = t - self._session_start_t
        ev = Event(kind=kind, t_abs=t, t_rel=rel, data=data)
        with self._lock:
            self._events.append(ev)
        return t

    def bar(self, idx: int) -> BarRecord:
        with self._lock:
            if idx not in self._bars:
                self._bars[idx] = BarRecord(bar_index=idx)
            return self._bars[idx]

    def all_events(self) -> List[Event]:
        with self._lock:
            return list(self._events)

    def all_bars(self) -> List[BarRecord]:
        with self._lock:
            return sorted(self._bars.values(), key=lambda b: b.bar_index)

    def record_error(self, code: int, msg: str) -> None:
        t = _now_ms()
        with self._lock:
            self._errors.append({"code": code, "msg": msg, "t": t})
        print(f"  [ERR {code}] {msg}")

    def to_json(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.all_events()],
            "bars": [
                {
                    "bar_index":    b.bar_index,
                    "ts":           f"{b.ts_num}/{b.ts_den}",
                    "bpm":          b.bpm,
                    "bar_dur_ms":   b.bar_dur_ms,
                    "notes_sent":   b.notes_sent,
                    "gen_latency_ms": b.gen_latency_ms,
                    "gen_total_ms": b.gen_total_ms,
                    "deadline_ms":  b.deadline_ms,
                    "slack_ms":     b.slack_ms,
                    "on_time":      b.on_time,
                    "gen_notes":    b.gen_notes,
                    "gen_dropped":  b.gen_dropped,
                }
                for b in self.all_bars()
            ],
            "errors": self._errors,
        }


# ── OSC endpoint (single UDP socket, send + receive) ─────────────────────────

class OscEndpoint:
    """
    One UDP socket bound to a local port, used for both sending to the server
    and receiving server replies.

    The server replies to the UDP source address of incoming packets.  By
    binding the client socket to a fixed port we ensure every reply arrives
    here, even as the OS chooses ephemeral ports for other sockets.
    """

    def __init__(self, local_port: int,
                 server_host: str, server_port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", local_port))
        self._sock.settimeout(0.5)
        self._server = (server_host, server_port)
        self._handlers: Dict[str, List] = {}
        self._default_handler = None
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="osc-recv")
        self._recv_thread.start()

    def send(self, address: str, *args) -> None:
        builder = OscMessageBuilder(address=address)
        for arg in args:
            if isinstance(arg, bool):
                builder.add_arg(int(arg), "i")
            elif isinstance(arg, int):
                builder.add_arg(arg, "i")
            elif isinstance(arg, float):
                builder.add_arg(arg, "f")
            elif isinstance(arg, str):
                builder.add_arg(arg, "s")
            else:
                builder.add_arg(arg)
        self._sock.sendto(builder.build().dgram, self._server)

    def on(self, address: str, callback) -> None:
        self._handlers.setdefault(address, []).append(callback)

    def set_default_handler(self, cb) -> None:
        self._default_handler = cb

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg  = OscMessage(data)
                addr = msg.address
                handled = False
                for pattern, cbs in list(self._handlers.items()):
                    if addr == pattern or addr.startswith(pattern):
                        for cb in cbs:
                            cb(addr, *msg)
                        handled = True
                if not handled and self._default_handler:
                    self._default_handler(addr, *msg)
            except Exception:
                pass


# ── MIDI data helpers ─────────────────────────────────────────────────────────

def load_piece(midi_path: str, resolution: int = 12) -> dict:
    """Parse MIDI → piece dict via midigpt C++ parser."""
    cfg = midigpt.TrainConfig()
    cfg.resolution = resolution
    cfg.num_bars   = 1
    cfg.min_tracks = 1
    raw = midigpt.midi_to_json_bytes(midi_path, cfg, "{}")
    if not raw:
        raise ValueError(f"Could not parse MIDI: {midi_path}")
    piece = json.loads(midigpt.json_bytes_to_string(raw))
    for track in piece.get("tracks", []):
        tt = track.get("track_type", 10)
        if tt == 8:
            track["track_type"] = 11
        elif tt not in (10, 11):
            track["track_type"] = 10
    return piece


def extract_notes(piece: dict, track_idx: int,
                  bar_idx: int) -> List[dict]:
    """
    Return sorted note-on dicts for (track_idx, bar_idx).
    Each: {pitch, velocity, onset (norm [0,1)), duration (norm)}.
    Note-offs (velocity=0) are excluded.
    """
    events_pool = piece.get("events", [])
    tracks = piece.get("tracks", [])
    if track_idx >= len(tracks):
        return []
    bars = tracks[track_idx].get("bars", [])
    if bar_idx >= len(bars):
        return []
    bar    = bars[bar_idx]
    ts_num = bar.get("ts_numerator",   4)
    ts_den = bar.get("ts_denominator", 4)
    res    = piece.get("resolution",  12)
    ticks  = int(ts_num * 4 * res / ts_den)
    if ticks == 0:
        ticks = 48

    notes = []
    for idx in bar.get("events", []):
        if idx >= len(events_pool):
            continue
        ev = events_pool[idx]
        if ev.get("velocity", 0) == 0:
            continue
        dur_ticks = ev.get("internal_duration", res)
        notes.append({
            "pitch":    ev["pitch"],
            "velocity": ev["velocity"],
            "onset":    ev["time"] / ticks,
            "duration": dur_ticks / ticks,
        })
    return sorted(notes, key=lambda n: n["onset"])


def get_bpm(piece: dict, default: float = 120.0) -> float:
    """Extract BPM from piece or return default."""
    if piece.get("tempo_changes"):
        qpm = piece["tempo_changes"][0].get("qpm", 0)
        if qpm > 0:
            return float(qpm)
    v = piece.get("tempo", 0)
    if v and v > 0:
        return float(v)
    return default


# ── Main simulation session ───────────────────────────────────────────────────

class OscSimSession:
    """
    Drives a full OSC session: init → play bars → stop.

    Runs MidiGPTServer in a background thread.  The simulation client uses a
    single UDP socket bound to response_port so the server always knows where
    to send generated notes.
    """

    def __init__(
        self,
        piece:            dict,
        num_human_tracks: int,
        server_port:      int,
        response_port:    int,
        agent_instrument: int,
        gen_params:       dict,
        ckpt:             str,
        bpm_override:     Optional[float],
        realtime_factor:  float,
        mode:             str,
        max_attempts:     int,
        diag:             DiagnosticsLog,
    ) -> None:
        self._piece      = piece
        self._num_human  = num_human_tracks
        self._gen_params = gen_params
        self._ckpt       = ckpt
        self._bpm        = bpm_override or get_bpm(piece)
        self._factor     = realtime_factor
        self._mode       = mode
        self._diag       = diag
        self._agent_inst = agent_instrument
        self._max_att    = max_attempts
        self._srv_port   = server_port
        self._rsp_port   = response_port

        self._ready_ev   = threading.Event()
        self._started_ev = threading.Event()

        # target_bar → abs time (ms) when generation was triggered
        self._gen_trigger_t: Dict[int, float] = {}

    # ── Server ────────────────────────────────────────────────────────────────

    def _start_server(self) -> None:
        self._server = MidiGPTServer(
            ckpt=self._ckpt,
            listen_port=self._srv_port,
            max_attempts=self._max_att,
        )
        for k, v in self._gen_params.items():
            if k in self._server._params:
                self._server._params[k] = v
        t = threading.Thread(
            target=lambda: self._server.serve("127.0.0.1"),
            daemon=True, name="osc-server")
        t.start()
        time.sleep(0.35)

    # ── OSC client ────────────────────────────────────────────────────────────

    def _start_client(self) -> None:
        self._ep = OscEndpoint(
            local_port=self._rsp_port,
            server_host="127.0.0.1",
            server_port=self._srv_port,
        )
        self._ep.on("/midigpt/session/ready",      self._on_ready)
        self._ep.on("/midigpt/session/started",    self._on_started)
        self._ep.on("/midigpt/generated/open",     self._on_gen_open)
        self._ep.on("/midigpt/generated/note",     self._on_gen_note)
        self._ep.on("/midigpt/generated/close",    self._on_gen_close)
        self._ep.on("/midigpt/generated/features", self._on_gen_features)
        self._ep.on("/midigpt/error",              self._on_error)
        self._ep.on("/midigpt/status",             self._on_status)
        self._ep.set_default_handler(self._on_unknown)

    # ── Response callbacks ────────────────────────────────────────────────────

    def _on_ready(self, addr, *args) -> None:
        self._diag.record("session_ready_recv")
        self._ready_ev.set()

    def _on_started(self, addr, *args) -> None:
        self._diag.record("session_started_recv")
        self._started_ev.set()

    def _on_gen_open(self, addr, *args) -> None:
        t = _now_ms()
        if len(args) < 3:
            return
        track_id   = int(args[0])
        bar_index  = int(args[1])
        note_count = int(args[2])
        self._diag.record("gen_open_recv",
                          track_id=track_id, bar_index=bar_index,
                          note_count=note_count)
        br = self._diag.bar(bar_index)
        br.gen_open_t = t
        lat = ""
        if bar_index in self._gen_trigger_t:
            lat = f"  latency={t - self._gen_trigger_t[bar_index]:.0f}ms"
        print(f"    ↳ gen/open   bar={bar_index:<3d}  note_count={note_count}{lat}")

    def _on_gen_note(self, addr, *args) -> None:
        if len(args) >= 2:
            self._diag.bar(int(args[1])).gen_notes += 1

    def _on_gen_close(self, addr, *args) -> None:
        t = _now_ms()
        if len(args) < 2:
            return
        track_id  = int(args[0])
        bar_index = int(args[1])
        br = self._diag.bar(bar_index)
        br.gen_close_t = t
        if bar_index in self._gen_trigger_t:
            br.gen_trigger_t = self._gen_trigger_t[bar_index]
        total = br.gen_total_ms
        slack = br.slack_ms
        flag  = ("✓" if (slack is not None and slack >= 0)
                 else ("✗" if slack is not None else "?"))
        slack_str = f"{slack:+.0f}ms" if slack is not None else "—"
        self._diag.record("gen_close_recv",
                          track_id=track_id, bar_index=bar_index,
                          total_ms=total, gen_notes=br.gen_notes)
        print(f"    ↳ gen/close  bar={bar_index:<3d}  total={total:.0f}ms"
              f"  slack={slack_str}  {flag}  ({br.gen_notes} notes)")

    def _on_gen_features(self, addr, *args) -> None:
        if len(args) < 7:
            return
        self._diag.record(
            "gen_features",
            bar_index=int(args[1]),
            note_density=float(args[2]),
            mean_pitch=float(args[3]),
            mean_velocity=float(args[4]),
            max_polyphony=int(args[5]),
            mean_duration=float(args[6]),
        )

    def _on_error(self, addr, *args) -> None:
        code = int(args[0]) if args else -1
        msg  = str(args[1]) if len(args) > 1 else ""
        self._diag.record_error(code, msg)

    def _on_status(self, addr, *args) -> None:
        state = str(args[0]) if args else "?"
        self._diag.record("status_recv", state=state)

    def _on_unknown(self, addr, *args) -> None:
        self._diag.record("unknown_recv", address=addr)

    # ── Session setup ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        print("  /session/init ...", end=" ", flush=True)
        self._diag.record("session_init_sent")
        self._ep.send("/midigpt/session/init", "osc-sim")
        if not self._ready_ev.wait(timeout=5.0):
            raise TimeoutError("No /session/ready within 5 s")
        print("ready")

        for i in range(self._num_human):
            t  = self._piece["tracks"][i]
            inst = t.get("instrument", 0)
            tt   = t.get("track_type", 10)
            self._ep.send("/midigpt/track/create", i, inst, tt, 0)
            print(f"  track {i}: inst={inst}  type={'drum' if tt==11 else 'melodic'}")

        agent_id = self._num_human
        self._ep.send("/midigpt/track/create",
                      agent_id, self._agent_inst, 10, 1)
        print(f"  agent:   inst={self._agent_inst}  id={agent_id}")

        p = self._gen_params
        self._ep.send("/midigpt/param/set", "buffer_bars",         p["buffer_bars"])
        self._ep.send("/midigpt/param/set", "lookahead_bars",      p["lookahead_bars"])
        self._ep.send("/midigpt/param/set", "model_dim",           p["model_dim"])
        self._ep.send("/midigpt/param/set", "num_anticipated_bars", p["num_anticipated_bars"])
        self._ep.send("/midigpt/param/set", "temperature",         float(p["temperature"]))

        print("  /session/start ...", end=" ", flush=True)
        self._diag.record("session_start_sent")
        self._ep.send("/midigpt/session/start")
        if not self._started_ev.wait(timeout=5.0):
            raise TimeoutError("No /session/started within 5 s")
        print("started")

    # ── Bar playback ──────────────────────────────────────────────────────────

    def _play_bar(self, bar_idx: int) -> None:
        # Gather notes from all conditioning tracks
        all_notes: List[Tuple[int, dict]] = []
        ts_num, ts_den = 4, 4
        for ti in range(self._num_human):
            bars = self._piece["tracks"][ti].get("bars", [])
            if bar_idx < len(bars):
                ts_num = bars[bar_idx].get("ts_numerator",   4)
                ts_den = bars[bar_idx].get("ts_denominator", 4)
            for note in extract_notes(self._piece, ti, bar_idx):
                all_notes.append((ti, note))

        bar_ms  = bar_duration_ms(ts_num, ts_den, self._bpm)
        sim_ms  = bar_ms / self._factor   # wall-clock duration
        t_start = _now_ms()

        br = self._diag.bar(bar_idx)
        br.ts_num  = ts_num
        br.ts_den  = ts_den
        br.bpm     = self._bpm
        br.start_t = t_start
        br._deadline_ms = bar_ms * self._gen_params["lookahead_bars"]

        if self._mode == "live":
            for track_id, note in sorted(all_notes, key=lambda x: x[1]["onset"]):
                target_t = t_start + note["onset"] * sim_ms
                wait_s   = (target_t - _now_ms()) / 1000.0
                if wait_s > 0:
                    time.sleep(wait_s)
                self._send_note(track_id, note, bar_idx)
            # Sleep until bar end
            wait_s = (t_start + sim_ms - _now_ms()) / 1000.0
            if wait_s > 0:
                time.sleep(wait_s)
        else:
            for track_id, note in all_notes:
                self._send_note(track_id, note, bar_idx)
            time.sleep(sim_ms / 1000.0)

        # Finalize bar
        t_end = _now_ms()
        br.end_t      = t_end
        br.notes_sent = sum(1 for _, n in all_notes)
        self._diag.record("bar_end_sent",
                          bar_index=bar_idx,
                          ts_num=ts_num, ts_den=ts_den,
                          notes_sent=br.notes_sent,
                          bar_ms=bar_ms, sim_ms=sim_ms)
        self._ep.send("/midigpt/bar/end", bar_idx, ts_num, ts_den)

        # Determine if this bar_end triggers generation and for which target
        k       = self._gen_params["lookahead_bars"]
        B       = self._gen_params["buffer_bars"]
        adapt   = self._gen_params.get("adapt_buffer", False)
        compl   = bar_idx + 1   # bars_completed after this bar

        target_bar = None
        if adapt:
            if compl + k >= B:
                target_bar = compl + k
        else:
            if compl >= B:
                target_bar = compl + k

        if target_bar is not None:
            t_trig = _now_ms()
            self._gen_trigger_t[target_bar] = t_trig
            self._diag.bar(target_bar).gen_trigger_t = t_trig
            self._diag.record("gen_trigger",
                              bar_index=bar_idx, target_bar=target_bar,
                              bars_completed=compl, t_trigger=t_trig)
            print(f"  bar {bar_idx:3d} | {ts_num}/{ts_den}"
                  f" | notes={br.notes_sent:2d}"
                  f" | sim_dur={sim_ms:.0f}ms"
                  f" | GEN→bar {target_bar}"
                  f"  deadline={br._deadline_ms:.0f}ms",
                  flush=True)
        else:
            buf_left = max(0, B - compl)
            print(f"  bar {bar_idx:3d} | {ts_num}/{ts_den}"
                  f" | notes={br.notes_sent:2d}"
                  f" | sim_dur={sim_ms:.0f}ms"
                  f" | buffer ({buf_left} bars left)",
                  flush=True)

    def _send_note(self, track_id: int, note: dict, bar_idx: int) -> None:
        self._diag.record("note_sent",
                          track_id=track_id, bar_index=bar_idx,
                          pitch=note["pitch"], velocity=note["velocity"],
                          onset=round(note["onset"], 4))
        self._ep.send(
            "/midigpt/note",
            track_id,
            int(note["pitch"]),
            int(note["velocity"]),
            float(note["onset"]),
            float(note["duration"]),
            bar_idx,
        )

    # ── Full session run ──────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "═" * 72)
        print("  MIDI-GPT OSC Simulation Session")
        print("═" * 72)

        print(f"\n[1/4] Starting server (port {self._srv_port}) …")
        self._start_server()

        print(f"[2/4] Starting client (response port {self._rsp_port}) …")
        self._start_client()

        print("[3/4] Session setup …")
        self._diag.session_start()
        self._setup()

        tracks    = self._piece.get("tracks", [])
        total_bars = len(tracks[0]["bars"]) if tracks else 0

        print(f"\n[4/4] Playback simulation")
        print(f"  MIDI:    {total_bars} bars, {self._num_human} conditioning tracks")
        print(f"  BPM:     {self._bpm:.1f}")
        print(f"  Mode:    {self._mode}   speed={self._factor:.1f}x")
        bdur = bar_duration_ms(4, 4, self._bpm)
        print(f"  Bar dur: {bdur:.0f} ms (4/4)  "
              f"→ sim {bdur/self._factor:.0f} ms/bar")
        print(f"  Gen:     buffer={self._gen_params['buffer_bars']}"
              f"  lookahead={self._gen_params['lookahead_bars']}"
              f"  model_dim={self._gen_params['model_dim']}"
              f"  j={self._gen_params['num_anticipated_bars']}")
        print(f"  Deadline per step: "
              f"{bdur * self._gen_params['lookahead_bars']:.0f} ms"
              f"  (lookahead * bar_dur)")
        print()

        for bar_idx in range(total_bars):
            self._play_bar(bar_idx)

        # Allow in-flight generation to finish
        print("\n  Waiting for in-flight generation …")
        time.sleep(1.0)

        self._diag.record("session_stop_sent")
        self._ep.send("/midigpt/session/stop")
        time.sleep(0.2)
        self._ep.close()
        print("  Done.")


# ── Diagnostics report ────────────────────────────────────────────────────────

def _fmt(v: Optional[float], unit: str = "ms", signed: bool = False) -> str:
    if v is None:
        return "—"
    fmt = f"{v:+.0f}" if signed else f"{v:.0f}"
    return fmt + unit


def print_report(diag: DiagnosticsLog) -> None:
    bars   = diag.all_bars()
    events = diag.all_events()

    print("\n" + "═" * 72)
    print("  DIAGNOSTICS REPORT")
    print("═" * 72)

    # ── Message-type counts ───────────────────────────────────────────────────
    kinds: Dict[str, int] = {}
    for ev in events:
        kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
    print("\nMessage counts:")
    for k in sorted(kinds):
        print(f"  {k:<30s}  {kinds[k]:5d}")

    # ── Per-bar table ─────────────────────────────────────────────────────────
    gen_bars = [b for b in bars if b.gen_trigger_t is not None]
    print(f"\nPer-bar generation summary  ({len(gen_bars)} generation steps):\n")
    HDR = (f"  {'Bar':>4}  {'TS':>5}  {'Sent':>4}  "
           f"{'Latency':>9}  {'Total':>9}  {'Deadline':>9}  "
           f"{'Slack':>9}  {'GenNotes':>8}  St")
    SEP = "  " + "─" * (len(HDR) - 2)
    print(HDR)
    print(SEP)
    for b in gen_bars:
        status = ("✓" if b.on_time
                  else ("✗" if b.on_time is not None else "?"))
        if b.gen_dropped:
            status = "DROP"
        print(f"  {b.bar_index:>4}  {b.ts_num}/{b.ts_den:<3}  {b.notes_sent:>4}  "
              f"{_fmt(b.gen_latency_ms):>9}  {_fmt(b.gen_total_ms):>9}  "
              f"{_fmt(b.deadline_ms):>9}  {_fmt(b.slack_ms,'ms',True):>9}  "
              f"{b.gen_notes:>8}  {status}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    latencies = [b.gen_total_ms for b in gen_bars if b.gen_total_ms is not None]
    slacks    = [b.slack_ms     for b in gen_bars if b.slack_ms     is not None]
    on_time   = [b for b in gen_bars if b.on_time is True]
    dropped   = [b for b in gen_bars if b.gen_dropped]
    total_gen_notes  = sum(b.gen_notes for b in bars)
    total_sent_notes = sum(b.notes_sent for b in bars)

    print(f"\nSummary")
    print(f"  Bars played           {len(bars)}")
    print(f"  Generation steps      {len(gen_bars)}")
    pct = 100 * len(on_time) / max(len(gen_bars), 1)
    print(f"  On-time               {len(on_time)}/{len(gen_bars)}  ({pct:.1f}%)")
    print(f"  Dropped (queue full)  {len(dropped)}")
    print(f"  Notes sent (cond.)    {total_sent_notes}")
    print(f"  Notes generated       {total_gen_notes}")

    if latencies:
        qs = quantiles(latencies, n=20) if len(latencies) >= 20 else None
        print(f"\nGeneration latency  (trigger → /generated/close)")
        print(f"  min     {min(latencies):.0f} ms")
        print(f"  mean    {mean(latencies):.0f} ms")
        print(f"  median  {median(latencies):.0f} ms")
        if qs:
            print(f"  p95     {qs[18]:.0f} ms")
        print(f"  max     {max(latencies):.0f} ms")

    if slacks:
        print(f"\nSlack  (deadline − latency, + = on-time)")
        print(f"  min     {min(slacks):+.0f} ms")
        print(f"  mean    {mean(slacks):+.0f} ms")
        print(f"  max     {max(slacks):+.0f} ms")

    # ── Compact event timeline ────────────────────────────────────────────────
    interesting = [e for e in events
                   if e.kind not in ("note_sent", "unknown_recv")]
    print(f"\nEvent timeline ({len(interesting)} non-note events,"
          f" {len(events)} total):")
    for ev in interesting[:120]:
        extra = "  ".join(
            f"{k}={v}" for k, v in ev.data.items()
            if k not in ("onset", "t_abs", "t_rel")
        )
        print(f"  {ev.t_rel:9.1f} ms  {ev.kind:<28s}  {extra}")
    if len(interesting) > 120:
        print(f"  … ({len(interesting) - 120} more — see --log_json)")

    print("\n" + "═" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate a live OSC session with pre-composed MIDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--midi",  required=True, help="Input MIDI file")
    p.add_argument("--ckpt",  required=True, help="Model checkpoint (.pt)")

    # Playback
    p.add_argument("--mode", choices=["live", "batched"], default="live",
                   help="live = notes at onset positions; "
                        "batched = burst at bar-start")
    p.add_argument("--realtime_factor", type=float, default=1.0,
                   help=">1 = faster than real-time (stress test)")
    p.add_argument("--bpm", type=float, default=None,
                   help="Override MIDI tempo (BPM).  Default: read from MIDI.")
    p.add_argument("--num_tracks", type=int, default=2,
                   help="Max conditioning tracks to take from the MIDI")
    p.add_argument("--agent_instrument", type=int, default=32,
                   help="MIDI program for agent track (32 = acoustic bass)")

    # Generation
    p.add_argument("--buffer",      type=int,   default=4, dest="buffer_bars")
    p.add_argument("--lookahead",   type=int,   default=2, dest="lookahead_bars")
    p.add_argument("--model_dim",   type=int,   default=8)
    p.add_argument("--j",           type=int,   default=1,
                   dest="num_anticipated_bars",
                   help="Bars generated per inference call")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_attempts",type=int,   default=5)
    p.add_argument("--adapt_buffer",action="store_true")

    # Ports
    p.add_argument("--server_port",   type=int, default=7400)
    p.add_argument("--response_port", type=int, default=7401)

    # Output
    p.add_argument("--log_json", type=str, default=None,
                   help="Write full event log to this JSON file")
    return p.parse_args()


def main() -> None:
    if not HAS_MIDIGPT:
        print("Error: midigpt not built.  Run scripts/compile_install.sh first.")
        sys.exit(1)
    if not HAS_PYTHONOSC:
        print("Error: python-osc not installed.  "
              "Run: pip install -e '.[osc]'")
        sys.exit(1)

    args = _parse_args()

    print(f"Loading MIDI: {args.midi}")
    try:
        piece = load_piece(args.midi)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    num_tracks = len(piece.get("tracks", []))
    num_human  = min(args.num_tracks, max(1, num_tracks - 1))
    total_bars = len(piece["tracks"][0]["bars"]) if num_tracks > 0 else 0
    bpm        = args.bpm or get_bpm(piece)
    print(f"  {num_tracks} tracks, {total_bars} bars, BPM≈{bpm:.1f}")

    if total_bars == 0:
        print("Error: MIDI has no bars.")
        sys.exit(1)
    if args.buffer_bars >= total_bars:
        print(f"Warning: buffer ({args.buffer_bars}) >= total_bars ({total_bars}). "
              "No generation will occur.")

    gen_params = {
        "buffer_bars":          args.buffer_bars,
        "lookahead_bars":       args.lookahead_bars,
        "model_dim":            args.model_dim,
        "num_anticipated_bars": args.num_anticipated_bars,
        "temperature":          args.temperature,
        "adapt_buffer":         args.adapt_buffer,
    }

    diag = DiagnosticsLog()
    session = OscSimSession(
        piece=piece,
        num_human_tracks=num_human,
        server_port=args.server_port,
        response_port=args.response_port,
        agent_instrument=args.agent_instrument,
        gen_params=gen_params,
        ckpt=args.ckpt,
        bpm_override=args.bpm,
        realtime_factor=args.realtime_factor,
        mode=args.mode,
        max_attempts=args.max_attempts,
        diag=diag,
    )

    try:
        session.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()

    print_report(diag)

    if args.log_json:
        out = Path(args.log_json)
        out.write_text(json.dumps(diag.to_json(), indent=2))
        print(f"\nFull event log → {out}")


if __name__ == "__main__":
    main()
