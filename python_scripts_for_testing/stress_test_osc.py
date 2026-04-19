"""
python_scripts_for_testing/stress_test_osc.py

Stress-test the MIDI-GPT OSC server with synthetically generated MIDI
(no real MIDI file required).  Exposes all generation controls and sweeps
multiple configurations to find the realtime performance boundary.

Usage examples
--------------
# Single run — 3 human tracks, 8 notes/bar, 1× speed:
python stress_test_osc.py --ckpt /path/to/model.pt \\
    --num_tracks 3 --notes_per_bar 8 --num_bars 16

# Full sweep — vary num_tracks vs notes_per_bar:
python stress_test_osc.py --ckpt /path/to/model.pt --sweep tracks

# Speed sweep — how fast can the model go?
python stress_test_osc.py --ckpt /path/to/model.pt --sweep speed

# Agent output density sweep — vary onset_density / polyphony controls:
python stress_test_osc.py --ckpt /path/to/model.pt --sweep agent_output

# All sweeps at once:
python stress_test_osc.py --ckpt /path/to/model.pt --sweep full

Output
------
Per-configuration table: mean_latency, p95_latency, on_time%, mean_notes_generated.
Realtime boundary: configurations above/below the threshold are marked.
"""

import argparse
import json
import os
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "python_scripts"))
sys.path.insert(0, str(_REPO / "build"))

try:
    from pythonosc.osc_message_builder import OscMessageBuilder
    from pythonosc.osc_message import OscMessage
    HAS_PYTHONOSC = True
except ImportError:
    HAS_PYTHONOSC = False

from osc_server import MidiGPTServer

# ── Helpers shared with simulate_osc_session ─────────────────────────────────

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def bar_duration_ms(ts_num: int, ts_den: int, bpm: float) -> float:
    quarter_notes_per_bar = ts_num * 4.0 / ts_den
    return quarter_notes_per_bar * 60_000.0 / bpm


# ── Synthetic piece generation ────────────────────────────────────────────────

# Melodic-friendly GM programs (piano family, strings, synths, etc.)
_MELODIC_PROGRAMS = [
    0,  1,  4,  5,  24, 25, 26, 27,  # piano, guitar
    32, 33, 34, 38,                   # bass
    40, 41, 42, 43, 44, 45, 46, 47,  # strings
    56, 57, 58, 60, 61,               # brass
    65, 66, 67, 68, 69, 70, 71,      # winds
    73, 74,                           # flute
    80, 81, 82, 83,                   # lead synths
    88, 89, 90,                       # pad synths
]

_GM_INST_NAMES: List[str] = [
    "acoustic_grand_piano", "bright_acoustic_piano", "electric_grand_piano",
    "honky_tonk_piano", "electric_piano_1", "electric_piano_2", "harpsichord",
    "clavi", "celesta", "glockenspiel", "music_box", "vibraphone", "marimba",
    "xylophone", "tubular_bells", "dulcimer", "drawbar_organ", "percussive_organ",
    "rock_organ", "church_organ", "reed_organ", "accordion", "harmonica",
    "tango_accordion", "acoustic_guitar_nylon", "acoustic_guitar_steel",
    "electric_guitar_jazz", "electric_guitar_clean", "electric_guitar_muted",
    "overdriven_guitar", "distortion_guitar", "guitar_harmonics", "acoustic_bass",
    "electric_bass_finger", "electric_bass_pick", "fretless_bass", "slap_bass_1",
    "slap_bass_2", "synth_bass_1", "synth_bass_2", "violin", "viola", "cello",
    "contrabass", "tremolo_strings", "pizzicato_strings", "orchestral_harp",
    "timpani", "string_ensemble_1", "string_ensemble_2", "synth_strings_1",
    "synth_strings_2", "choir_aahs", "voice_oohs", "synth_voice", "orchestra_hit",
    "trumpet", "trombone", "tuba", "muted_trumpet", "french_horn", "brass_section",
    "synth_brass_1", "synth_brass_2", "soprano_sax", "alto_sax", "tenor_sax",
    "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "piccolo",
    "flute", "recorder", "pan_flute", "blown_bottle", "shakuhachi", "whistle",
    "ocarina", "lead_1_square", "lead_2_sawtooth", "lead_3_calliope",
    "lead_4_chiff", "lead_5_charang", "lead_6_voice", "lead_7_fifths",
    "lead_8_bass__lead", "pad_1_new_age", "pad_2_warm", "pad_3_polysynth",
    "pad_4_choir", "pad_5_bowed", "pad_6_metallic", "pad_7_halo", "pad_8_sweep",
    "fx_1_rain", "fx_2_soundtrack", "fx_3_crystal", "fx_4_atmosphere",
    "fx_5_brightness", "fx_6_goblins", "fx_7_echoes", "fx_8_sci_fi",
    "sitar", "banjo", "shamisen", "koto", "kalimba", "bag_pipe", "fiddle",
    "shanai", "tinkle_bell", "agogo", "steel_drums", "woodblock", "taiko_drum",
    "melodic_tom", "synth_drum", "reverse_cymbal", "guitar_fret_noise",
    "breath_noise", "seashore", "bird_tweet", "telephone_ring", "helicopter",
    "applause", "gunshot",
]


def _gm_name(program: int) -> str:
    if 0 <= program < len(_GM_INST_NAMES):
        return _GM_INST_NAMES[program]
    return "any"


def generate_random_piece(
    num_tracks:    int   = 2,
    num_bars:      int   = 16,
    notes_per_bar: int   = 8,
    resolution:    int   = 12,
    ts_num:        int   = 4,
    ts_den:        int   = 4,
    bpm:           float = 120.0,
    pitch_lo:      int   = 48,
    pitch_hi:      int   = 84,
    vel_lo:        int   = 64,
    vel_hi:        int   = 100,
    max_polyphony: int   = 3,
    dur_lo:        float = 0.1,   # normalized duration fraction of bar
    dur_hi:        float = 0.5,
    seed:          Optional[int] = None,
    random_instruments: bool = True,
) -> Tuple[dict, List[int]]:
    """
    Build a synthetic piece dict in the flat-pool format expected by
    PieceState / simulate_osc_session.  No MIDI file needed.

    Returns (piece_dict, instruments) where instruments[i] is the MIDI
    program number for track i.
    """
    rng = random.Random(seed)

    # Choose instruments randomly (from melodic pool) or sequentially
    instruments: List[int] = []
    for i in range(num_tracks):
        if random_instruments:
            prog = rng.choice(_MELODIC_PROGRAMS)
        else:
            prog = _MELODIC_PROGRAMS[i % len(_MELODIC_PROGRAMS)]
        instruments.append(prog)

    ticks = int(ts_num * 4 * resolution / ts_den)

    events_pool: List[dict] = []
    tracks_data: List[dict] = []

    for ti in range(num_tracks):
        bars_data: List[dict] = []
        for bi in range(num_bars):
            # How many notes actually land in this bar
            n = rng.randint(max(1, notes_per_bar // 2), notes_per_bar)
            bar_event_indices: List[int] = []

            # Track active note endpoints for polyphony limiting
            active_end_ticks: List[int] = []

            for _ in range(n):
                onset_tick = rng.randint(0, ticks - 1)
                dur_ticks  = int(rng.uniform(dur_lo, dur_hi) * ticks)
                dur_ticks  = max(1, min(dur_ticks, ticks - onset_tick))
                end_tick   = onset_tick + dur_ticks

                # Polyphony check: count notes active at onset_tick
                active_end_ticks = [e for e in active_end_ticks if e > onset_tick]
                if len(active_end_ticks) >= max_polyphony:
                    # Skip this note rather than violate polyphony limit
                    continue
                active_end_ticks.append(end_tick)

                pitch    = rng.randint(pitch_lo, pitch_hi)
                velocity = rng.randint(vel_lo, vel_hi)

                ev = {
                    "pitch":             pitch,
                    "velocity":          velocity,
                    "time":              onset_tick,
                    "internal_duration": dur_ticks,
                }
                pool_idx = len(events_pool)
                events_pool.append(ev)
                bar_event_indices.append(pool_idx)

            bars_data.append({
                "ts_numerator":   ts_num,
                "ts_denominator": ts_den,
                "events":         sorted(bar_event_indices,
                                         key=lambda i: events_pool[i]["time"]),
            })

        tracks_data.append({
            "instrument": instruments[ti],
            "track_type": 10,  # STANDARD_MIDI_TRACK
            "bars":       bars_data,
        })

    piece = {
        "resolution":    resolution,
        "tempo":         bpm,
        "tempo_changes": [{"qpm": bpm}],
        "tracks":        tracks_data,
        "events":        events_pool,
    }
    return piece, instruments


# ── Agent controls dataclass ───────────────────────────────────────────────────

@dataclass
class AgentControls:
    """
    Per-agent generation controls sent via /midigpt/track/param/set.
    Defaults match _AGENT_PARAM_DEFAULTS in realtime_state.py (0 = "off").

    polyphony_hard_limit is a *computational guardrail*, not a musical control.
    It should always be set to a reasonably high value (e.g., 8) to bound the
    sampling search space without artistically constraining the output.
    Musical controls are: onset_density (drums), min/max_polyphony_q and
    min/max_note_duration_q (instruments), min/max_pitch, key_signature.
    """
    onset_density:        int   = 0   # BarLevelOnsetDensityLevel (0–5), primary drum control
    polyphony_hard_limit: int   = 8   # computational guardrail — always set, keep ≥6
    min_polyphony_q:      int   = 0   # PolyphonyLevel low bound  (0 = off)
    max_polyphony_q:      int   = 0   # PolyphonyLevel high bound (0 = off)
    min_note_duration_q:  int   = 0   # NoteDurationLevel low bound
    max_note_duration_q:  int   = 0   # NoteDurationLevel high bound
    min_pitch:            int   = 0   # MIDI pitch floor (0 = off)
    max_pitch:            int   = 127 # MIDI pitch ceiling
    key_signature:        int   = 0   # 0=C, 1=G, …, 11=B (0 = unconstrained)

    def label(self) -> str:
        """One-line human-readable description of non-default musical params."""
        parts = []
        if self.onset_density:
            parts.append(f"density={self.onset_density}")
        if self.min_polyphony_q or self.max_polyphony_q:
            parts.append(f"poly_q=[{self.min_polyphony_q},{self.max_polyphony_q}]")
        if self.min_note_duration_q or self.max_note_duration_q:
            parts.append(f"dur_q=[{self.min_note_duration_q},{self.max_note_duration_q}]")
        if self.min_pitch or self.max_pitch != 127:
            parts.append(f"pitch=[{self.min_pitch},{self.max_pitch}]")
        if self.key_signature:
            parts.append(f"key={self.key_signature}")
        return ",".join(parts) if parts else "default"


# ── Sweep configuration ────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """One test configuration."""
    label:           str
    num_tracks:      int   = 2
    notes_per_bar:   int   = 8
    num_bars:        int   = 16
    realtime_factor: float = 1.0
    bpm:             float = 120.0
    ts_num:          int   = 4
    ts_den:          int   = 4
    model_dim:       int   = 4
    j:               int   = 1          # num_anticipated_bars
    buffer_bars:     int   = 4
    lookahead_bars:  int   = 2
    temperature:     float = 1.0
    agent_controls:  AgentControls = field(default_factory=AgentControls)
    max_polyphony:   int   = 3          # input polyphony cap
    dur_lo:          float = 0.1
    dur_hi:          float = 0.5
    pitch_lo:        int   = 48
    pitch_hi:        int   = 84
    random_inst:     bool  = True
    seed:            Optional[int] = 42


def _note_sweeps() -> List[SweepConfig]:
    """Vary notes per bar (input density)."""
    return [
        SweepConfig(label=f"notes={n}", notes_per_bar=n)
        for n in [2, 4, 8, 16, 24, 32]
    ]


def _track_sweeps() -> List[SweepConfig]:
    """Vary number of conditioning tracks."""
    return [
        SweepConfig(label=f"tracks={t}", num_tracks=t)
        for t in [1, 2, 3, 4, 5]
    ]


def _speed_sweeps() -> List[SweepConfig]:
    """Vary realtime factor — higher = less wall time per bar."""
    return [
        SweepConfig(label=f"speed={s}x", realtime_factor=s)
        for s in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    ]


def _model_dim_sweeps() -> List[SweepConfig]:
    """Vary model_dim (context window size)."""
    return [
        SweepConfig(label=f"model_dim={d}", model_dim=d)
        for d in [1, 2, 4, 8]
    ]


def _j_sweeps() -> List[SweepConfig]:
    """Vary bars generated per inference call."""
    return [
        SweepConfig(label=f"j={j}", j=j)
        for j in [1, 2, 4]
    ]


def _tempo_sweeps() -> List[SweepConfig]:
    """Vary tempo — affects bar duration available for generation."""
    return [
        SweepConfig(label=f"bpm={b}", bpm=b)
        for b in [60, 90, 120, 160, 200]
    ]


def _ts_sweeps() -> List[SweepConfig]:
    """Vary time signature — changes bar length."""
    return [
        SweepConfig(label=f"ts={n}/{d}", ts_num=n, ts_den=d)
        for (n, d) in [(3, 4), (4, 4), (6, 8), (5, 4), (7, 8)]
    ]


def _agent_output_sweeps() -> List[SweepConfig]:
    """
    Vary musical attribute controls. polyphony_hard_limit stays at its default
    (8, computational guardrail) throughout — it is never a swept variable.

    Instrument controls: min/max_polyphony_q, min/max_note_duration_q, pitch range, key.
    Drum control: onset_density.
    """
    def AC(**kw) -> AgentControls:
        """AgentControls with guardrail always present."""
        return AgentControls(**kw)  # polyphony_hard_limit=8 is the class default

    configs = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    configs.append(SweepConfig(label="agent=default"))

    # ── onset_density (drums / overall density) ───────────────────────────────
    configs.append(SweepConfig(label="agent=density_1",
                               agent_controls=AC(onset_density=1)))
    configs.append(SweepConfig(label="agent=density_3",
                               agent_controls=AC(onset_density=3)))
    configs.append(SweepConfig(label="agent=density_5",
                               agent_controls=AC(onset_density=5)))

    # ── Polyphony quantile range (instruments) ────────────────────────────────
    # Sparse: level 1 only (mostly single notes)
    configs.append(SweepConfig(label="agent=poly_q[1,1]",
                               agent_controls=AC(min_polyphony_q=1,
                                                 max_polyphony_q=1)))
    # Medium: levels 2–3
    configs.append(SweepConfig(label="agent=poly_q[2,3]",
                               agent_controls=AC(min_polyphony_q=2,
                                                 max_polyphony_q=3)))
    # Dense: levels 4–5
    configs.append(SweepConfig(label="agent=poly_q[4,5]",
                               agent_controls=AC(min_polyphony_q=4,
                                                 max_polyphony_q=5)))

    # ── Note duration quantile range (instruments) ────────────────────────────
    # Short: staccato-ish
    configs.append(SweepConfig(label="agent=dur_q[1,2]",
                               agent_controls=AC(min_note_duration_q=1,
                                                 max_note_duration_q=2)))
    # Medium
    configs.append(SweepConfig(label="agent=dur_q[3,4]",
                               agent_controls=AC(min_note_duration_q=3,
                                                 max_note_duration_q=4)))
    # Long: legato / held notes
    configs.append(SweepConfig(label="agent=dur_q[5,6]",
                               agent_controls=AC(min_note_duration_q=5,
                                                 max_note_duration_q=6)))

    # ── Pitch range ───────────────────────────────────────────────────────────
    configs.append(SweepConfig(label="agent=pitch_low[36,60]",
                               agent_controls=AC(min_pitch=36, max_pitch=60)))
    configs.append(SweepConfig(label="agent=pitch_mid[48,72]",
                               agent_controls=AC(min_pitch=48, max_pitch=72)))
    configs.append(SweepConfig(label="agent=pitch_high[60,84]",
                               agent_controls=AC(min_pitch=60, max_pitch=84)))

    # ── Key signature ─────────────────────────────────────────────────────────
    configs.append(SweepConfig(label="agent=key_C",
                               agent_controls=AC(key_signature=0)))   # C major
    configs.append(SweepConfig(label="agent=key_G",
                               agent_controls=AC(key_signature=1)))   # G major
    configs.append(SweepConfig(label="agent=key_Fs",
                               agent_controls=AC(key_signature=6)))   # F# major

    # ── Combinations ─────────────────────────────────────────────────────────
    # Sparse texture: low density + monophonic-ish + short durations
    configs.append(SweepConfig(label="agent=sparse",
                               agent_controls=AC(onset_density=1,
                                                 min_polyphony_q=1,
                                                 max_polyphony_q=1,
                                                 min_note_duration_q=1,
                                                 max_note_duration_q=2)))
    # Dense texture: high density + chordal + legato
    configs.append(SweepConfig(label="agent=dense",
                               agent_controls=AC(onset_density=5,
                                                 min_polyphony_q=3,
                                                 max_polyphony_q=5,
                                                 min_note_duration_q=4,
                                                 max_note_duration_q=6)))
    # Melodic: medium density, strictly monophonic, mid-pitch, C major
    configs.append(SweepConfig(label="agent=melodic",
                               agent_controls=AC(onset_density=3,
                                                 min_polyphony_q=1,
                                                 max_polyphony_q=1,
                                                 min_pitch=60, max_pitch=84,
                                                 key_signature=0)))

    return configs


def _full_sweeps() -> List[SweepConfig]:
    """Grid cross of the most diagnostic axes."""
    configs = []
    for t in [1, 2, 3]:
        for n in [4, 8, 16]:
            for s in [1.0, 4.0]:
                label = f"trk={t},notes={n},spd={s}x"
                configs.append(SweepConfig(
                    label=label, num_tracks=t,
                    notes_per_bar=n, realtime_factor=s))
    return configs


def _regression_sweeps() -> List[SweepConfig]:
    """
    Grid over (bpm, notes_per_bar) to fit a response-time plane.
    """
    bpm_values   = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    notes_values = [2, 4, 8, 12, 16, 24, 32]
    configs = []
    for b in bpm_values:
        for n in notes_values:
            configs.append(SweepConfig(
                label=f"bpm={b},notes={n}",
                bpm=b,
                notes_per_bar=n,
            ))
    return configs


def _regression_v2_sweeps() -> List[SweepConfig]:
    """
    Expanded 3-way grid: BPM × time-signature × notes_per_bar.
    Time signature changes bar_ms independently of BPM, giving much richer
    coverage of the bar-duration axis and separating its effect from tempo.
    """
    bpm_values = [60, 75, 90, 105, 120, 140, 160, 180, 200, 220, 240]
    ts_values  = [(3, 4), (4, 4), (5, 4), (6, 8), (7, 8)]
    notes_values = [2, 4, 8, 16, 24, 32]
    configs = []
    for b in bpm_values:
        for (ts_n, ts_d) in ts_values:
            for n in notes_values:
                configs.append(SweepConfig(
                    label=f"bpm={b},ts={ts_n}/{ts_d},notes={n}",
                    bpm=b,
                    ts_num=ts_n,
                    ts_den=ts_d,
                    notes_per_bar=n,
                ))
    return configs


SWEEP_PRESETS: Dict[str, List[SweepConfig]] = {
    "notes":        _note_sweeps,
    "tracks":       _track_sweeps,
    "speed":        _speed_sweeps,
    "model_dim":    _model_dim_sweeps,
    "j":            _j_sweeps,
    "tempo":        _tempo_sweeps,
    "ts":           _ts_sweeps,
    "agent_output": _agent_output_sweeps,
    "regression":   _regression_sweeps,
    "regression_v2": _regression_v2_sweeps,
    "full":         _full_sweeps,
}


# ── OSC helpers (reused from simulate_osc_session) ───────────────────────────

class OscEndpoint:
    def __init__(self, local_port: int, server_host: str,
                 server_port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", local_port))
        self._sock.settimeout(0.5)
        self._server = (server_host, server_port)
        self._handlers: Dict[str, list] = {}
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

    def on(self, address: str, cb) -> None:
        self._handlers.setdefault(address, []).append(cb)

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


# ── Per-run result ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    cfg:               SweepConfig
    latencies_ms:      List[float]   = field(default_factory=list)
    totals_ms:         List[float]   = field(default_factory=list)
    slacks_ms:         List[float]   = field(default_factory=list)
    notes_generated:   List[int]     = field(default_factory=list)
    on_time_count:     int           = 0
    gen_count:         int           = 0
    error_count:       int           = 0

    @property
    def on_time_pct(self) -> Optional[float]:
        return (100.0 * self.on_time_count / self.gen_count
                if self.gen_count > 0 else None)

    @property
    def mean_latency(self) -> Optional[float]:
        return mean(self.latencies_ms) if self.latencies_ms else None

    @property
    def p95_latency(self) -> Optional[float]:
        if len(self.latencies_ms) < 2:
            return self.latencies_ms[0] if self.latencies_ms else None
        s = sorted(self.latencies_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def mean_notes(self) -> Optional[float]:
        return mean(self.notes_generated) if self.notes_generated else None

    @property
    def max_notes(self) -> Optional[int]:
        return max(self.notes_generated) if self.notes_generated else None

    @property
    def mean_slack(self) -> Optional[float]:
        return mean(self.slacks_ms) if self.slacks_ms else None

    def bar_dur_ms(self) -> float:
        return bar_duration_ms(self.cfg.ts_num, self.cfg.ts_den, self.cfg.bpm)


# ── One stress-test run ────────────────────────────────────────────────────────

class StressRun:
    """
    Runs one SweepConfig against a live MidiGPTServer instance, then
    returns a RunResult.  Instantiates its own server (on a given port).
    """

    def __init__(self, cfg: SweepConfig, ckpt: str,
                 server_port: int, response_port: int,
                 max_attempts: int, verbose: bool) -> None:
        self._cfg         = cfg
        self._ckpt        = ckpt
        self._srv_port    = server_port
        self._rsp_port    = response_port
        self._max_att     = max_attempts
        self._verbose     = verbose
        self._result      = RunResult(cfg=cfg)

        # Events for sequencing
        self._ready_ev    = threading.Event()
        self._started_ev  = threading.Event()

        # Generation tracking
        self._gen_trigger_t: Dict[int, float] = {}
        self._gen_open_t:    Dict[int, float] = {}
        self._pending_notes: Dict[int, int]   = {}  # bar → count so far
        self._lock           = threading.Lock()

    def run(self) -> RunResult:
        if not HAS_PYTHONOSC:
            raise RuntimeError("python-osc not installed; run: pip install python-osc")

        # Generate synthetic piece
        piece, instruments = generate_random_piece(
            num_tracks    = self._cfg.num_tracks,
            num_bars      = self._cfg.num_bars,
            notes_per_bar = self._cfg.notes_per_bar,
            ts_num        = self._cfg.ts_num,
            ts_den        = self._cfg.ts_den,
            bpm           = self._cfg.bpm,
            pitch_lo      = self._cfg.pitch_lo,
            pitch_hi      = self._cfg.pitch_hi,
            max_polyphony = self._cfg.max_polyphony,
            dur_lo        = self._cfg.dur_lo,
            dur_hi        = self._cfg.dur_hi,
            seed          = self._cfg.seed,
            random_instruments = self._cfg.random_inst,
        )
        self._piece       = piece
        self._instruments = instruments

        self._start_server()
        self._start_client()
        try:
            self._setup()
            self._playback()
            self._ep.send("/midigpt/session/stop")
            time.sleep(0.3)  # let final generation complete
        finally:
            self._ep.close()
        return self._result

    # ── Server ────────────────────────────────────────────────────────────────

    def _start_server(self) -> None:
        self._server = MidiGPTServer(
            ckpt=self._ckpt,
            listen_port=self._srv_port,
            max_attempts=self._max_att,
        )
        # Inject global params before serve() starts the OSC loop
        gp = self._server._params
        gp["buffer_bars"]         = self._cfg.buffer_bars
        gp["lookahead_bars"]      = self._cfg.lookahead_bars
        gp["model_dim"]           = self._cfg.model_dim
        gp["num_anticipated_bars"] = self._cfg.j
        gp["temperature"]         = self._cfg.temperature
        t = threading.Thread(
            target=lambda: self._server.serve("127.0.0.1"),
            daemon=True, name=f"srv-{self._srv_port}")
        t.start()
        time.sleep(0.35)

    # ── Client ────────────────────────────────────────────────────────────────

    def _start_client(self) -> None:
        self._ep = OscEndpoint(
            local_port  = self._rsp_port,
            server_host = "127.0.0.1",
            server_port = self._srv_port,
        )
        self._ep.on("/midigpt/session/ready",      self._on_ready)
        self._ep.on("/midigpt/session/started",    self._on_started)
        self._ep.on("/midigpt/generated/open",     self._on_gen_open)
        self._ep.on("/midigpt/generated/note",     self._on_gen_note)
        self._ep.on("/midigpt/generated/close",    self._on_gen_close)
        self._ep.on("/midigpt/generated/features", self._on_gen_features)
        self._ep.on("/midigpt/error",              self._on_error)

    def _on_ready(self, addr, *args) -> None:
        self._ready_ev.set()

    def _on_started(self, addr, *args) -> None:
        self._started_ev.set()

    def _on_gen_open(self, addr, *args) -> None:
        t = _now_ms()
        if len(args) < 2:
            return
        bar_index = int(args[1])
        with self._lock:
            self._gen_open_t[bar_index] = t
            trig = self._gen_trigger_t.get(bar_index)
        if trig is not None:
            lat = t - trig
            self._result.latencies_ms.append(lat)
            if self._verbose:
                print(f"      gen/open  bar={bar_index}  lat={lat:.0f}ms")

    def _on_gen_note(self, addr, *args) -> None:
        if len(args) >= 2:
            bi = int(args[1])
            with self._lock:
                self._pending_notes[bi] = self._pending_notes.get(bi, 0) + 1

    def _on_gen_close(self, addr, *args) -> None:
        t = _now_ms()
        if len(args) < 2:
            return
        bar_index = int(args[1])
        with self._lock:
            trig  = self._gen_trigger_t.get(bar_index)
            notes = self._pending_notes.pop(bar_index, 0)

        self._result.notes_generated.append(notes)
        self._result.gen_count += 1

        if trig is not None:
            total = t - trig
            self._result.totals_ms.append(total)
            # Deadline = lookahead × bar_duration (in musical time)
            bar_ms    = bar_duration_ms(self._cfg.ts_num, self._cfg.ts_den, self._cfg.bpm)
            deadline  = self._cfg.lookahead_bars * bar_ms
            slack     = deadline - total
            self._result.slacks_ms.append(slack)
            if slack >= 0:
                self._result.on_time_count += 1
            if self._verbose:
                flag = "✓" if slack >= 0 else "✗"
                print(f"      gen/close bar={bar_index}  total={total:.0f}ms"
                      f"  slack={slack:+.0f}ms  {flag}  ({notes} notes)")

    def _on_gen_features(self, addr, *args) -> None:
        pass  # captured in RunResult via gen_notes per bar

    def _on_error(self, addr, *args) -> None:
        code = int(args[0]) if args else -1
        msg  = str(args[1]) if len(args) > 1 else ""
        self._result.error_count += 1
        if self._verbose:
            print(f"      [ERR {code}] {msg}")

    # ── Session setup ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._ep.send("/midigpt/session/init", "stress-test")
        if not self._ready_ev.wait(timeout=10.0):
            raise TimeoutError("No /session/ready within 10s")

        for i, prog in enumerate(self._instruments):
            self._ep.send("/midigpt/track/create", i, prog, 10, 0)

        agent_id = len(self._instruments)
        agent_prog = random.choice(_MELODIC_PROGRAMS)
        self._ep.send("/midigpt/track/create", agent_id, agent_prog, 10, 1)

        # Send agent parameter controls
        ac = self._cfg.agent_controls
        ctrl_params = {
            "onset_density":        ac.onset_density,
            "polyphony_hard_limit": ac.polyphony_hard_limit,
            "min_polyphony_q":      ac.min_polyphony_q,
            "max_polyphony_q":      ac.max_polyphony_q,
            "min_note_duration_q":  ac.min_note_duration_q,
            "max_note_duration_q":  ac.max_note_duration_q,
            "min_pitch":            ac.min_pitch,
            "max_pitch":            ac.max_pitch,
            "key_signature":        ac.key_signature,
        }
        for name, val in ctrl_params.items():
            self._ep.send("/midigpt/track/param/set",
                          agent_id, name, int(val))

        self._ep.send("/midigpt/session/start")
        if not self._started_ev.wait(timeout=10.0):
            raise TimeoutError("No /session/started within 10s")

    # ── Bar playback ──────────────────────────────────────────────────────────

    def _playback(self) -> None:
        ticks    = int(self._cfg.ts_num * 4 * 12 / self._cfg.ts_den)
        bar_ms   = bar_duration_ms(self._cfg.ts_num, self._cfg.ts_den, self._cfg.bpm)
        sim_ms   = bar_ms / self._cfg.realtime_factor

        k = self._cfg.lookahead_bars
        B = self._cfg.buffer_bars

        for bi in range(self._cfg.num_bars):
            t_start = _now_ms()

            # Burst-send all notes for all conditioning tracks at bar start
            events_pool = self._piece["events"]
            for ti in range(self._cfg.num_tracks):
                bars = self._piece["tracks"][ti].get("bars", [])
                if bi >= len(bars):
                    continue
                bar_data = bars[bi]
                for idx in bar_data.get("events", []):
                    if idx >= len(events_pool):
                        continue
                    ev = events_pool[idx]
                    if ev.get("velocity", 0) == 0:
                        continue
                    onset_norm = ev["time"] / max(ticks, 1)
                    dur_norm   = ev.get("internal_duration", 1) / max(ticks, 1)
                    self._ep.send(
                        "/midigpt/note",
                        ti,
                        ev["pitch"],
                        ev["velocity"],
                        float(onset_norm),
                        float(dur_norm),
                        bi,
                    )

            # Sleep to simulate bar duration
            elapsed = _now_ms() - t_start
            remaining = sim_ms - elapsed
            if remaining > 0:
                time.sleep(remaining / 1000.0)

            self._ep.send("/midigpt/bar/end",
                          bi, self._cfg.ts_num, self._cfg.ts_den)

            # Track generation trigger timing
            completed = bi + 1
            if completed >= B:
                target_bar = completed + k
                with self._lock:
                    if target_bar not in self._gen_trigger_t:
                        self._gen_trigger_t[target_bar] = _now_ms()

            if self._verbose:
                print(f"    bar {bi:3d} | {self._cfg.ts_num}/{self._cfg.ts_den}"
                      f" | sim_dur={sim_ms:.0f}ms")


# ── Report printing ────────────────────────────────────────────────────────────

def print_run_result(r: RunResult, verbose: bool = True) -> None:
    """Print summary for one run."""
    bar_ms = r.bar_dur_ms()
    deadline_ms = r.cfg.lookahead_bars * bar_ms

    print(f"\n  Config: {r.cfg.label}")
    print(f"    tracks={r.cfg.num_tracks}  notes/bar={r.cfg.notes_per_bar}"
          f"  bars={r.cfg.num_bars}  speed={r.cfg.realtime_factor}x"
          f"  bpm={r.cfg.bpm}  ts={r.cfg.ts_num}/{r.cfg.ts_den}"
          f"  model_dim={r.cfg.model_dim}  j={r.cfg.j}")
    print(f"    agent: {r.cfg.agent_controls.label()}")
    print(f"    bar_dur={bar_ms:.0f}ms  deadline={deadline_ms:.0f}ms"
          f"  (lookahead={r.cfg.lookahead_bars})")

    if r.gen_count == 0:
        print("    ⚠ No generations completed.")
        return

    print(f"    generations: {r.gen_count}  errors: {r.error_count}"
          f"  on-time: {r.on_time_count}/{r.gen_count}"
          f" ({r.on_time_pct:.0f}%)")

    if r.latencies_ms:
        print(f"    latency (trigger→open):  "
              f"mean={mean(r.latencies_ms):.0f}ms  "
              f"p95={r.p95_latency:.0f}ms")
    if r.totals_ms:
        print(f"    total   (trigger→close): "
              f"mean={mean(r.totals_ms):.0f}ms  "
              f"max={max(r.totals_ms):.0f}ms  "
              f"min={min(r.totals_ms):.0f}ms")
    if r.slacks_ms:
        print(f"    slack:  mean={mean(r.slacks_ms):+.0f}ms  "
              f"min={min(r.slacks_ms):+.0f}ms")
    if r.notes_generated:
        print(f"    notes generated/bar:  "
              f"mean={r.mean_notes:.1f}  max={r.max_notes}  "
              f"total={sum(r.notes_generated)}")


def print_sweep_table(results: List[RunResult]) -> None:
    """Print a comparison table for all sweep results."""
    col_w = [28, 8, 8, 8, 10, 8, 8, 10]
    headers = ["Config", "on-time%", "mean_lat", "p95_lat",
               "mean_total", "slack", "mean_notes", "errors"]
    sep = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + "=" * len(sep))
    print("SWEEP RESULTS")
    print("=" * len(sep))
    print(sep)
    print("-" * len(sep))

    for r in results:
        on_pct   = f"{r.on_time_pct:.0f}%" if r.on_time_pct is not None else "—"
        m_lat    = f"{mean(r.latencies_ms):.0f}" if r.latencies_ms else "—"
        p95_lat  = f"{r.p95_latency:.0f}"    if r.latencies_ms else "—"
        m_total  = f"{mean(r.totals_ms):.0f}" if r.totals_ms   else "—"
        m_slack  = (f"{mean(r.slacks_ms):+.0f}" if r.slacks_ms  else "—")
        m_notes  = f"{r.mean_notes:.1f}" if r.notes_generated else "—"
        errs     = str(r.error_count)

        # Realtime marker
        rt_ok = (r.on_time_pct is not None and r.on_time_pct >= 95.0)
        marker = " ✓" if rt_ok else (" ~" if r.on_time_pct is not None and r.on_time_pct >= 50 else " ✗")
        row_label = (r.cfg.label + marker)[:col_w[0]]

        row = "  ".join(v.ljust(w) for v, w in zip(
            [row_label, on_pct, m_lat, p95_lat, m_total, m_slack, m_notes, errs],
            col_w,
        ))
        print(row)

    print("-" * len(sep))
    print("  ✓ = on-time ≥95%  ~ = on-time 50–94%  ✗ = on-time <50%")
    print("  All times in ms.  Slack = deadline − total (positive = on time).")
    print("  deadline = lookahead_bars × bar_duration_ms\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test MIDI-GPT OSC server with synthetic MIDI.")

    # Required
    p.add_argument("--ckpt", required=True,
                   help="Path to TorchScript checkpoint (.pt)")

    # Sweep mode
    p.add_argument("--sweep",
                   choices=list(SWEEP_PRESETS.keys()),
                   default=None,
                   help="Run a systematic sweep instead of a single config. "
                        "Options: " + ", ".join(SWEEP_PRESETS))

    # Single-run piece parameters
    p.add_argument("--num_tracks",    type=int,   default=2,
                   help="Number of conditioning tracks (default: 2)")
    p.add_argument("--notes_per_bar", type=int,   default=8,
                   help="Target notes per bar per track (default: 8)")
    p.add_argument("--num_bars",      type=int,   default=16,
                   help="Bars to play (default: 16)")
    p.add_argument("--bpm",           type=float, default=120.0,
                   help="Tempo in BPM (default: 120)")
    p.add_argument("--ts",            type=str,   default="4/4",
                   help="Time signature as N/D (default: 4/4)")
    p.add_argument("--max_polyphony", type=int,   default=3,
                   help="Max simultaneous notes in synthetic input (default: 3)")
    p.add_argument("--pitch_lo",      type=int,   default=48)
    p.add_argument("--pitch_hi",      type=int,   default=84)
    p.add_argument("--dur_lo",        type=float, default=0.1,
                   help="Min note duration as fraction of bar (default: 0.1)")
    p.add_argument("--dur_hi",        type=float, default=0.5,
                   help="Max note duration as fraction of bar (default: 0.5)")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Random seed for piece generation (default: 42)")
    p.add_argument("--random_instruments", action="store_true", default=True,
                   help="Randomize track instruments (default: true)")

    # Generation controls
    p.add_argument("--realtime_factor", type=float, default=1.0,
                   help="Speed multiplier >1 = faster than real-time (default: 1)")
    p.add_argument("--model_dim",       type=int,   default=4,
                   help="Model context window in bars (default: 4)")
    p.add_argument("--j",               type=int,   default=1,
                   help="Bars generated per inference call (default: 1)")
    p.add_argument("--buffer",          type=int,   default=4,
                   help="Buffer bars before agent starts (default: 4)")
    p.add_argument("--lookahead",       type=int,   default=2,
                   help="Lookahead bars (default: 2)")
    p.add_argument("--temperature",     type=float, default=1.0)

    # Agent controls
    p.add_argument("--onset_density",        type=int, default=0,
                   help="Drum/density control 0–5 (0=off)")
    p.add_argument("--polyphony_hard_limit",  type=int, default=8,
                   help="Computational guardrail — keep ≥6 (default: 8)")
    p.add_argument("--min_polyphony_q",       type=int, default=0)
    p.add_argument("--max_polyphony_q",       type=int, default=0)
    p.add_argument("--min_note_duration_q",   type=int, default=0)
    p.add_argument("--max_note_duration_q",   type=int, default=0)
    p.add_argument("--min_pitch_agent",       type=int, default=0)
    p.add_argument("--max_pitch_agent",       type=int, default=127)
    p.add_argument("--key_signature",         type=int, default=0)

    # Infrastructure
    p.add_argument("--server_port",   type=int, default=7400)
    p.add_argument("--response_port", type=int, default=7401)
    p.add_argument("--max_attempts",  type=int, default=3)
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--log_json",      type=str, default=None,
                   help="Save all results as JSON to this path")

    return p.parse_args()


def main() -> None:
    if not HAS_PYTHONOSC:
        print("ERROR: python-osc not installed.  Run: pip install python-osc")
        sys.exit(1)

    args = parse_args()

    # Parse time signature
    ts_parts = args.ts.split("/")
    ts_num, ts_den = int(ts_parts[0]), int(ts_parts[1])

    # Build agent controls from CLI
    agent_ctrl = AgentControls(
        onset_density        = args.onset_density,
        polyphony_hard_limit = args.polyphony_hard_limit,
        min_polyphony_q      = args.min_polyphony_q,
        max_polyphony_q      = args.max_polyphony_q,
        min_note_duration_q  = args.min_note_duration_q,
        max_note_duration_q  = args.max_note_duration_q,
        min_pitch            = args.min_pitch_agent,
        max_pitch            = args.max_pitch_agent,
        key_signature        = args.key_signature,
    )

    if args.sweep:
        # Run all configurations in the chosen preset
        preset_fn = SWEEP_PRESETS[args.sweep]
        configs = preset_fn() if callable(preset_fn) else preset_fn
        print(f"\nRunning sweep '{args.sweep}' — {len(configs)} configs")
        print(f"  ckpt:    {args.ckpt}")
        print(f"  bars:    {args.num_bars}")
        print(f"  buffer:  {args.buffer}  lookahead: {args.lookahead}")
        print()

        results: List[RunResult] = []
        port_offset = 0
        for cfg in configs:
            # Override bars and infrastructure params from CLI
            cfg.num_bars      = args.num_bars
            cfg.buffer_bars   = args.buffer
            cfg.lookahead_bars = args.lookahead
            cfg.max_polyphony = args.max_polyphony

            srv_port = args.server_port  + port_offset * 2
            rsp_port = args.response_port + port_offset * 2
            port_offset += 1

            print(f"[{port_offset}/{len(configs)}] {cfg.label} …", flush=True)
            try:
                run = StressRun(
                    cfg=cfg, ckpt=args.ckpt,
                    server_port=srv_port, response_port=rsp_port,
                    max_attempts=args.max_attempts, verbose=args.verbose)
                result = run.run()
                results.append(result)
                if args.verbose:
                    print_run_result(result)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append(RunResult(cfg=cfg, error_count=1))

            # Brief pause between runs to avoid port conflicts
            time.sleep(0.5)

        print_sweep_table(results)

        if args.log_json:
            data = [
                {
                    "label":          r.cfg.label,
                    "num_tracks":     r.cfg.num_tracks,
                    "notes_per_bar":  r.cfg.notes_per_bar,
                    "realtime_factor":r.cfg.realtime_factor,
                    "bpm":            r.cfg.bpm,
                    "ts":             f"{r.cfg.ts_num}/{r.cfg.ts_den}",
                    "model_dim":      r.cfg.model_dim,
                    "j":              r.cfg.j,
                    "on_time_pct":    r.on_time_pct,
                    "mean_latency_ms":r.mean_latency,
                    "p95_latency_ms": r.p95_latency,
                    "mean_total_ms":  mean(r.totals_ms) if r.totals_ms else None,
                    "mean_slack_ms":  r.mean_slack,
                    "mean_notes_gen": r.mean_notes,
                    "max_notes_gen":  r.max_notes,
                    "gen_count":      r.gen_count,
                    "error_count":    r.error_count,
                }
                for r in results
            ]
            with open(args.log_json, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to {args.log_json}")

    else:
        # Single run
        cfg = SweepConfig(
            label           = "single",
            num_tracks      = args.num_tracks,
            notes_per_bar   = args.notes_per_bar,
            num_bars        = args.num_bars,
            realtime_factor = args.realtime_factor,
            bpm             = args.bpm,
            ts_num          = ts_num,
            ts_den          = ts_den,
            model_dim       = args.model_dim,
            j               = args.j,
            buffer_bars     = args.buffer,
            lookahead_bars  = args.lookahead,
            temperature     = args.temperature,
            agent_controls  = agent_ctrl,
            max_polyphony   = args.max_polyphony,
            dur_lo          = args.dur_lo,
            dur_hi          = args.dur_hi,
            pitch_lo        = args.pitch_lo,
            pitch_hi        = args.pitch_hi,
            random_inst     = args.random_instruments,
            seed            = args.seed,
        )
        print(f"\nSingle stress run")
        print(f"  ckpt: {args.ckpt}")
        print(f"  {cfg.num_tracks} tracks × {cfg.notes_per_bar} notes/bar"
              f" × {cfg.num_bars} bars  @{cfg.bpm}bpm {cfg.ts_num}/{cfg.ts_den}"
              f"  {cfg.realtime_factor}× speed")
        print()

        run = StressRun(
            cfg=cfg, ckpt=args.ckpt,
            server_port=args.server_port, response_port=args.response_port,
            max_attempts=args.max_attempts, verbose=True)
        result = run.run()
        print_run_result(result, verbose=True)

        if args.log_json:
            with open(args.log_json, "w") as f:
                json.dump({
                    "label":         cfg.label,
                    "latencies_ms":  result.latencies_ms,
                    "totals_ms":     result.totals_ms,
                    "slacks_ms":     result.slacks_ms,
                    "notes_generated": result.notes_generated,
                    "on_time_pct":   result.on_time_pct,
                    "gen_count":     result.gen_count,
                    "error_count":   result.error_count,
                }, f, indent=2)
            print(f"Results saved to {args.log_json}")


if __name__ == "__main__":
    main()
