#!/usr/bin/env python3
"""
step_viz.py — Real-time agent simulation with per-step generation visualization.

For every generation step prints:
  1. FULL-PIECE GRID   — every bar × track, colour-coded by role.
  2. WINDOW ZOOM       — the D-bar model window in detail, with bar-level annotations.
  3. TOKEN VIEW        — structural tokens the model actually sees (no note-level detail):
                          TRACK / BAR[Nev|MASK|emp] / >>>GEN>>> / TRACK_END

Run:
    python python_scripts_for_testing/step_viz.py \\
        --midi tests/short_midi/Aicha.mid \\
        --ckpt /scratch/triana24/.midigpt/models/ghost_baseline_340k.pt \\
        [--buffer 4] [--lookahead 1] [--num_anticipated_bars 1] [--model_dim 8] \\
        [--mask_gap] [--adapt_buffer] [--no_infer] [--max_steps N]

Skip inference (grid-only dry run):
    python step_viz.py --midi ... --no_infer
"""

import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import time
import torch
torch.set_num_threads(1)

try:
    import midigpt
except ImportError:
    print("ERROR: midigpt not installed — run 'pip install -e .[train]'")
    sys.exit(1)

_GM_MAPPING = {
    0: "acoustic_grand_piano", 1: "bright_acoustic_piano", 2: "electric_grand_piano", 3: "honky_tonk_piano",
    4: "electric_piano_1", 5: "electric_piano_2", 6: "harpsichord", 7: "clavi",
    8: "celesta", 9: "glockenspiel", 10: "music_box", 11: "vibraphone", 12: "marimba", 13: "xylophone",
    14: "tubular_bells", 15: "dulcimer", 16: "drawbar_organ", 17: "percussive_organ", 18: "rock_organ",
    19: "church_organ", 20: "reed_organ", 21: "accordion", 22: "harmonica", 23: "tango_accordion",
    24: "acoustic_guitar_nylon", 25: "acoustic_guitar_steel", 26: "electric_guitar_jazz",
    27: "electric_guitar_clean", 28: "electric_guitar_muted", 29: "overdriven_guitar",
    30: "distortion_guitar", 31: "guitar_harmonics", 32: "acoustic_bass", 33: "electric_bass_finger",
    34: "electric_bass_pick", 35: "fretless_bass", 36: "slap_bass_1", 37: "slap_bass_2",
    38: "synth_bass_1", 39: "synth_bass_2", 40: "violin", 41: "viola", 42: "cello", 43: "contrabass",
    44: "tremolo_strings", 45: "pizzicato_strings", 46: "orchestral_harp", 47: "timpani",
    48: "string_ensemble_1", 49: "string_ensemble_2", 50: "synth_strings_1", 51: "synth_strings_2",
    52: "choir_aahs", 53: "voice_oohs", 54: "synth_voice", 55: "orchestra_hit", 56: "trumpet",
    57: "trombone", 58: "tuba", 59: "muted_trumpet", 60: "french_horn", 61: "brass_section",
    62: "synth_brass_1", 63: "synth_brass_2", 64: "soprano_sax", 65: "alto_sax", 66: "tenor_sax",
    67: "baritone_sax", 68: "oboe", 69: "english_horn", 70: "bassoon", 71: "clarinet", 72: "piccolo",
    73: "flute", 74: "recorder", 75: "pan_flute", 76: "blown_bottle", 77: "shakuhachi", 78: "whistle",
    79: "ocarina", 80: "lead_1_square", 81: "lead_2_sawtooth", 82: "lead_3_calliope", 83: "lead_4_chiff",
    84: "lead_5_charang", 85: "lead_6_voice", 86: "lead_7_fifths", 87: "lead_8_bass__lead",
    88: "pad_1_new_age", 89: "pad_2_warm", 90: "pad_3_polysynth", 91: "pad_4_choir", 92: "pad_5_bowed",
    93: "pad_6_metallic", 94: "pad_7_halo", 95: "pad_8_sweep", 96: "fx_1_rain", 97: "fx_2_soundtrack",
    98: "fx_3_crystal", 99: "fx_4_atmosphere", 100: "fx_5_brightness", 101: "fx_6_goblins",
    102: "fx_7_echoes", 103: "fx_8_sci_fi", 104: "sitar", 105: "banjo", 106: "shamisen", 107: "koto",
    108: "kalimba", 109: "bag_pipe", 110: "fiddle", 111: "shanai", 112: "tinkle_bell", 113: "agogo",
    114: "steel_drums", 115: "woodblock", 116: "taiko_drum", 117: "melodic_tom", 118: "synth_drum",
    119: "reverse_cymbal", 120: "guitar_fret_noise", 121: "breath_noise", 122: "seashore",
    123: "bird_tweet", 124: "telephone_ring", 125: "helicopter", 126: "applause", 127: "gunshot"
}

# ── ANSI colour helpers ───────────────────────────────────────────────────────

RST  = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
GRY  = "\033[90m"
RED  = "\033[91m"
GRN  = "\033[92m"
YEL  = "\033[93m"
BLU  = "\033[94m"
MAG  = "\033[95m"
CYN  = "\033[96m"
WHT  = "\033[97m"

def _c(color: str, text: str) -> str:
    return color + text + RST

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# ── Bar roles ─────────────────────────────────────────────────────────────────

# Roles assigned to each (track, bar) cell in the grid.
#
#   OUT         : bar is outside the D-bar model window entirely.
#   CTX_EV      : in window, human conditioning bar with events (before playhead).
#   CTX_EMPTY   : in window, human conditioning bar, no events yet.
#   MASKED      : in window, future=True → TOKEN_MASK_BAR token.
#   AGENT_EV    : in window, agent prefix bar with previously generated events.
#   AGENT_EMPTY : in window, agent prefix bar, still empty (no events).
#   GEN_KEEP    : generation target [t, t+j) — piece_insert writes these back.
#   GEN_DISC    : model generates these (status_rehighlight expansion) but
#                 piece_insert discards them — only occurs in the early phase
#                 when the window is anchored at 0 and t+j-1 < window_end.

OUT         = "out"
CTX_EV      = "ctx_ev"
CTX_EMPTY   = "ctx_empty"
MASKED      = "masked"
AGENT_EV    = "agent_ev"
AGENT_EMPTY = "agent_empty"
GEN_KEEP    = "gen_keep"
GEN_DISC    = "gen_disc"

# (short-label, 3-char-glyph, colour) per role
_ROLE_META = {
    OUT:         ("  ",   "  ·",   GRY),
    CTX_EV:      ("CE",   "███",   GRN),
    CTX_EMPTY:   ("C-",   "░░░",   DIM + GRN),
    MASKED:      ("MM",   "▓▓▓",   YEL),
    AGENT_EV:    ("AE",   "▪▪▪",   BLU),
    AGENT_EMPTY: ("A-",   "───",   DIM + BLU),
    GEN_KEEP:    ("GK",   "◆◆◆",   BOLD + CYN),
    GEN_DISC:    ("GD",   "✕✕✕",   RED),
}

def _glyph(role: str) -> str:
    _, g, col = _ROLE_META[role]
    return col + g + RST

def _label(role: str) -> str:
    lbl, _, col = _ROLE_META[role]
    return col + lbl + RST

def _legend_line() -> str:
    parts = [
        _c(GRN,        "███ ctx-events"),
        _c(DIM+GRN,    "░░░ ctx-empty"),
        _c(YEL,        "▓▓▓ MASK(TOKEN_MASK_BAR)"),
        _c(BLU,        "▪▪▪ agent-pfx-events"),
        _c(DIM+BLU,    "─── agent-pfx-empty"),
        _c(BOLD+CYN,   "◆◆◆ GEN-KEEP[t,t+j)"),
        _c(RED,        "✕✕✕ GEN-DISCARD(early)"),
        _c(GRY,        " ·  outside-window"),
    ]
    return "  ".join(parts)


# ── Role computation ──────────────────────────────────────────────────────────

def compute_window(target_bar: int, j: int, D: int, total_bars: int):
    """Return (t_start, window_end) inclusive bar indices for the D-bar window."""
    t_start     = max(0, (target_bar + j - 1) - D + 1)
    window_end  = min(t_start + D - 1, total_bars - 1)
    return t_start, window_end


def assign_role(
    track_idx: int,
    bar_idx:   int,
    num_human: int,
    playhead:  int,
    target_bar: int,
    j:         int,
    t_start:   int,
    window_end: int,
    mask_gap:  bool,
    has_events: bool,
) -> str:
    in_window = t_start <= bar_idx <= window_end
    is_agent  = (track_idx == num_human)

    if not in_window:
        return OUT

    if is_agent:
        if bar_idx >= target_bar + j:
            # status_rehighlight expands suffix-AR to window_end inside sample_step,
            # but piece_insert only writes [t, t+j). This bar is generated then discarded.
            return GEN_DISC
        if bar_idx >= target_bar:
            return GEN_KEEP
        # Before target_bar: agent prefix
        if mask_gap and playhead <= bar_idx < target_bar:
            return MASKED       # gap masked → TOKEN_MASK_BAR
        return AGENT_EV if has_events else AGENT_EMPTY

    else:   # human track
        if bar_idx >= playhead:
            return MASKED       # future → TOKEN_MASK_BAR
        return CTX_EV if has_events else CTX_EMPTY


# ── Grid rendering ────────────────────────────────────────────────────────────

def _bar_header(total_bars: int, t_start: int, window_end: int, cell: int = 3) -> str:
    """Return two lines: one bar-index ruler and one window-bracket line."""
    sep = " "
    nums = sep.join(f"{b:^{cell}}" for b in range(total_bars))
    # Window bracket
    brk = [" " * cell] * total_bars
    brk[t_start]    = "┌" + "─" * (cell - 1)
    brk[window_end] = "─" * (cell - 1) + "┐"
    for b in range(t_start + 1, window_end):
        brk[b] = "─" * cell
    bracket = sep.join(brk)
    return (
        _c(GRY, f"bar: {nums}") + "\n" +
        _c(GRY, f"     {bracket}")
    )


def render_full_grid(
    piece:      dict,
    num_human:  int,
    playhead:   int,
    target_bar: int,
    j:          int,
    t_start:    int,
    window_end: int,
    mask_gap:   bool,
    total_bars: int,
) -> str:
    lines = []
    cell  = 3
    sep   = " "

    lines.append(_bar_header(total_bars, t_start, window_end, cell))

    for ti in range(num_human + 1):  # +1 for agent
        is_agent = (ti == num_human)
        label    = (f"Agent   " if is_agent else f"Human{ti} ")[:8]
        row = []
        for b in range(total_bars):
            ev  = len(piece["tracks"][ti]["bars"][b].get("events", [])) > 0
            rol = assign_role(ti, b, num_human, playhead, target_bar, j,
                              t_start, window_end, mask_gap, ev)
            row.append(_glyph(rol))
        lines.append(f"  {_c(BOLD, label)} {sep.join(row)}")

    # Window legend row
    pad_left  = t_start * (cell + len(sep))
    win_width = (window_end - t_start + 1) * (cell + len(sep)) - len(sep)
    win_label = f"{'window [' + str(t_start) + ',' + str(window_end) + ']':^{win_width}}"
    lines.append("  " + " " * 8 + " " * pad_left + _c(GRY, win_label))

    return "\n".join(lines)


def render_window_zoom(
    piece:      dict,
    num_human:  int,
    playhead:   int,
    target_bar: int,
    j:          int,
    t_start:    int,
    window_end: int,
    mask_gap:   bool,
) -> str:
    """Render the D-bar window with event counts below each bar."""
    lines = []
    cell  = 5  # wider cells for zoom
    sep   = " "
    D     = window_end - t_start + 1

    # Bar index header (relative to t_start and absolute)
    bar_abs  = sep.join(f"{b:^{cell}}"           for b in range(t_start, window_end + 1))
    bar_rel  = sep.join(f"{'(+'+str(b-t_start)+')':^{cell}}" for b in range(t_start, window_end + 1))
    lines.append(f"  {'bar':>8} {_c(GRY, bar_abs)}")
    lines.append(f"  {'':>8} {_c(GRY, bar_rel)}")

    for ti in range(num_human + 1):
        is_agent = (ti == num_human)
        label    = (f"Agent   " if is_agent else f"Human{ti} ")[:8]

        glyphs  = []
        annots  = []  # event counts / labels beneath each cell
        for b in range(t_start, window_end + 1):
            nev = len(piece["tracks"][ti]["bars"][b].get("events", []))
            ev  = nev > 0
            rol = assign_role(ti, b, num_human, playhead, target_bar, j,
                              t_start, window_end, mask_gap, ev)
            _, g, col = _ROLE_META[rol]
            glyphs.append(col + f" {g} " + RST)   # 5-char cell

            if rol == OUT:
                annots.append(f"{'':^{cell}}")
            elif rol == MASKED:
                annots.append(_c(YEL, f"{'MASK':^{cell}}"))
            elif rol == GEN_KEEP:
                annots.append(_c(BOLD+CYN, f"{'GEN':^{cell}}"))
            elif rol == GEN_DISC:
                annots.append(_c(RED, f"{'DISC':^{cell}}"))
            elif is_agent and rol in (AGENT_EV, AGENT_EMPTY):
                annots.append(_c(BLU, f"{'pfx':^{cell}}"))
            else:
                tag = f"{nev}ev" if ev else "emp"
                annots.append(_c(GRN, f"{tag:^{cell}}"))

        lines.append(f"  {_c(BOLD, label)} {sep.join(glyphs)}")
        lines.append(f"  {'':>8} {sep.join(annots)}")

    # Mark t and t+j-1
    def _ptr(b: int, label: str, col: str) -> str:
        pos = (b - t_start) * (cell + len(sep))
        return " " * 10 + " " * pos + _c(col, label)

    lines.append(_ptr(target_bar,         f"▲t={target_bar}", BOLD+CYN))
    if j > 1:
        lines.append(_ptr(target_bar+j-1, f"▲t+j-1={target_bar+j-1}", CYN))
    if window_end > target_bar + j - 1:
        lines.append(
            _ptr(target_bar + j,
                 f"← DISC starts here (early phase, window_end={window_end})", RED)
        )

    return "\n".join(lines)


# ── Token view ────────────────────────────────────────────────────────────────

def render_token_view(
    piece:      dict,
    num_human:  int,
    playhead:   int,
    target_bar: int,
    t_start:    int,
    window_end: int,
    mask_gap:   bool,
) -> str:
    """
    Show structural tokens per track for the window slice.

    Human tracks: TRACK · [BAR·b[Nev] or BAR·b[MASK]] · ... · TRACK_END
    Agent track:  TRACK · [BAR·b[pfx] or BAR·b[MASK]] · ... · >>>GENERATE>>> · TRACK_END

    The >>>GENERATE>>> marker is where the prompt is truncated and
    the model begins auto-regressive generation.
    """
    lines = []

    def _tok(s: str, col: str = "") -> str:
        return (col + s + RST) if col else s

    for ti in range(num_human + 1):
        is_agent = (ti == num_human)
        label    = (f"Agent   " if is_agent else f"Human{ti} ")[:8]
        tokens   = []

        tokens.append(_tok("TRACK", BOLD))

        for b in range(t_start, window_end + 1):
            nev = len(piece["tracks"][ti]["bars"][b].get("events", []))
            ev  = nev > 0
            rol = assign_role(ti, b, num_human, playhead, target_bar, b,
                              t_start, window_end, mask_gap, ev)
            # Re-compute properly (b here is bar index, j not needed for tok view)
            rol = assign_role(ti, b, num_human, playhead, target_bar,
                              1,   # j=1 for display purposes (doesn't matter here)
                              t_start, window_end, mask_gap, ev)

            # For agent: prompt ends at target_bar; mark the break
            if is_agent and b == target_bar:
                tokens.append(_tok("◄━━ GENERATE ━━►", BOLD + CYN))
                break

            if rol == MASKED:
                tokens.append(_tok(f"BAR·{b}[MASK]", YEL))
            elif is_agent and rol in (AGENT_EV, AGENT_EMPTY):
                tag = f"{nev}ev" if ev else "emp"
                tokens.append(_tok(f"BAR·{b}[{tag}]", BLU))
            elif rol == CTX_EV:
                tokens.append(_tok(f"BAR·{b}[{nev}ev]", GRN))
            elif rol == CTX_EMPTY:
                tokens.append(_tok(f"BAR·{b}[emp]", DIM + GRN))
            else:
                tokens.append(_tok(f"BAR·{b}[?]", GRY))

        if not is_agent:
            tokens.append(_tok("TRACK_END", BOLD))

        lines.append(f"  {_c(BOLD, label)} {_tok('│', GRY)} {_tok(' · ', GRY).join(tokens)}")

    return "\n".join(lines)


# ── Token sequence extraction ────────────────────────────────────────────────

def build_input_tokens(
    piece:       dict,
    num_human:   int,
    agent_idx:   int,
    playhead:    int,
    target_bar:  int,
    j:           int,
    t_start:     int,
    window_end:  int,
    mask_gap:    bool,
    enc,
    agent_min_dur=None,
    agent_max_dur=None,
    agent_min_poly=None,
    agent_max_poly=None,
    no_masking=False,
):
    """
    Return (prompt_tokens, all_tokens, trunc_idx) where:
      prompt_tokens  — tokens the model actually receives as input (up to the
                       first BAR token of target_bar in the agent track).
      all_tokens     — the full windowed sequence (prompt + masked remainder).
      trunc_idx      — index in all_tokens where truncation happens.

    Masking follows the same rules as the C++ status:
      human tracks  : bars >= playhead           → future=True → TOKEN_MASK_BAR
      agent track   : bars >= target_bar + j     → future=True → TOKEN_MASK_BAR
                      bars in [playhead, target_bar) with mask_gap → future=True
    """
    # Build windowed piece with future flags applied
    windowed = dict(piece)
    windowed_tracks = []
    for ti, track in enumerate(piece["tracks"]):
        is_agent = (ti == agent_idx)
        tc = dict(track)
        bars = []
        for b in range(t_start, window_end + 1):
            bar = copy.deepcopy(track["bars"][b])
            if is_agent:
                if no_masking:
                    bar["future"] = False
                else:
                    bar["future"] = (b >= target_bar + j) or (
                        mask_gap and playhead <= b < target_bar
                    )
            else:
                if no_masking:
                    bar["future"] = False
                else:
                    bar["future"] = (b >= playhead)
            bars.append(bar)
        tc["bars"] = bars
        windowed_tracks.append(tc)
    windowed["tracks"] = windowed_tracks

    all_tokens = enc.json_to_tokens(json.dumps(windowed))

    # Find truncation point: first TOKEN_BAR of (target_bar - t_start)
    # in the agent track (last track in windowed_tracks).
    target_bar_rel = target_bar - t_start
    trunc_idx      = len(all_tokens)
    track_idx      = -1
    bar_idx        = -1
    rep            = enc.rep

    for i, tok in enumerate(all_tokens):
        if rep.is_token_type(tok, midigpt.TOKEN_TYPE.TRACK):
            track_idx += 1
            bar_idx    = -1
        elif rep.is_token_type(tok, midigpt.TOKEN_TYPE.TRACK_END):
            pass
        elif rep.is_token_type(tok, midigpt.TOKEN_TYPE.BAR):
            bar_idx += 1
            if track_idx == agent_idx and bar_idx == target_bar_rel:
                trunc_idx = i
                break

    prompt = all_tokens[:trunc_idx]
    prompt = override_agent_attribute_tokens(
        prompt, agent_idx, enc,
        agent_min_dur, agent_max_dur,
        agent_min_poly, agent_max_poly
    )
    return prompt


def override_agent_attribute_tokens(tokens, agent_idx, enc, min_dur, max_dur, min_poly, max_poly):
    """
    Replace the agent track's TOKEN_MIN_NOTE_DURATION / TOKEN_MAX_NOTE_DURATION
    and TOKEN_MIN_POLYPHONY / TOKEN_MAX_POLYPHONY in a prompt token list.
    """
    if all(v is None for v in (min_dur, max_dur, min_poly, max_poly)):
        return tokens
    result = list(tokens)
    track_idx = -1
    rep = enc.rep
    for i, tok in enumerate(result):
        if rep.is_token_type(tok, midigpt.TOKEN_TYPE.TRACK):
            track_idx += 1
        elif track_idx == agent_idx:
            if min_dur is not None and min_dur > 0 and rep.is_token_type(tok, midigpt.TOKEN_TYPE.MIN_NOTE_DURATION):
                result[i] = rep.encode_partial(midigpt.TOKEN_TYPE.MIN_NOTE_DURATION, min_dur - 1)
            elif max_dur is not None and max_dur > 0 and rep.is_token_type(tok, midigpt.TOKEN_TYPE.MAX_NOTE_DURATION):
                result[i] = rep.encode_partial(midigpt.TOKEN_TYPE.MAX_NOTE_DURATION, max_dur - 1)
            elif min_poly is not None and min_poly > 0 and rep.is_token_type(tok, midigpt.TOKEN_TYPE.MIN_POLYPHONY):
                result[i] = rep.encode_partial(midigpt.TOKEN_TYPE.MIN_POLYPHONY, min_poly - 1)
            elif max_poly is not None and max_poly > 0 and rep.is_token_type(tok, midigpt.TOKEN_TYPE.MAX_POLYPHONY):
                result[i] = rep.encode_partial(midigpt.TOKEN_TYPE.MAX_POLYPHONY, max_poly - 1)
    return result


def parse_schedule(val, n_steps):
    """Accept int, 'a,b,c' or None. Returns list of length n_steps (cycling if shorter)."""
    if val is None:
        return [None] * n_steps
    try:
        items = [int(x) for x in str(val).split(",")]
        return [items[i % len(items)] for i in range(n_steps)]
    except Exception:
        return [None] * n_steps


def derive_controls_from_piece(piece, num_human):
    """
    Estimate polyphony and duration levels from human tracks so the agent
    gets controls that are in the same ballpark as the conditioning content.
    Returns (min_poly_status, max_poly_status, min_dur_status, max_dur_status)
    as 1-indexed status values (0 = no override).
    """
    poly_vals, dur_vals = [], []
    for ti in range(num_human):
        track = piece["tracks"][ti]
        for bar in track.get("bars", []):
            ev = bar.get("events", [])
            n = len(ev)
            if n > 0:
                poly_vals.append(n)
    if not poly_vals:
        return 2, 4, 3, 5  # mono–4note, 8th–quarter
    poly_vals.sort()
    p15 = poly_vals[max(0, int(len(poly_vals) * 0.15))]
    p85 = poly_vals[min(len(poly_vals)-1, int(len(poly_vals) * 0.85))]
    # rough: events per bar ÷ 4 ≈ polyphony level (capped 0-9)
    poly_min = max(1, min(9, p15 // 4))
    poly_max = max(poly_min+1, min(9, p85 // 4))
    # default durations: 8th note min, quarter note max
    return poly_min + 1, poly_max + 1, 3, 5


def _dur_level(d_ticks):
    """Match attribute_control.h: clip(log2(d/3)+1, 0, 5). resolution=24."""
    import math
    return int(min(5, max(0, math.log2(max(d_ticks / 3.0, 1e-6)) + 1)))


def _pair_note_durations(bar_ev_indices, all_events):
    """
    Pair note-ons with their note-offs by pitch to get note durations.
    Generated events have {time, velocity, pitch} — no internalDuration.
    Falls back to internalDuration if present (pre-computed pieces).
    Returns list of (pitch, start_tick, duration_ticks) for note-ons.
    """
    evs = [all_events[i] for i in bar_ev_indices if i < len(all_events)]
    note_ons = [(e["pitch"], e["time"], e.get("internalDuration", None))
                for e in evs if e.get("velocity", 0) > 0]
    note_offs = [(e["pitch"], e["time"])
                 for e in evs if e.get("velocity", 0) == 0]

    notes = []
    for pitch, t_on, int_dur in note_ons:
        if int_dur is not None:
            notes.append((pitch, t_on, int_dur))
        else:
            # find nearest matching note-off
            matches = [(t_off - t_on) for (p, t_off) in note_offs
                       if p == pitch and t_off > t_on]
            dur = min(matches) if matches else 12  # default: 8th note
            notes.append((pitch, t_on, dur))
    return notes


def compute_bar_controls(bar_events, all_events):
    """
    Compute per-bar polyphony and duration levels for the eval.
    polyphony_q: clip(max_simultaneous, 1, 10) - 1  (matches C++ per-bar value)
    duration_q: 15th/85th percentile of duration levels across all notes in bar
    Returns dict for one bar. Track-level quantile eval is done by
    compute_track_controls() across all bars.
    """
    if not bar_events:
        return {"poly_max": 0, "poly_q": 0, "min_dur_q": 0, "max_dur_q": 0, "n_notes": 0}
    notes = _pair_note_durations(bar_events, all_events)
    if not notes:
        return {"poly_max": 0, "poly_q": 0, "min_dur_q": 0, "max_dur_q": 0, "n_notes": 0}

    # polyphony: max simultaneous notes (sweep-line)
    tl = []
    for _, t_on, dur in notes:
        tl.append((t_on, +1))
        tl.append((t_on + max(1, dur), -1))
    tl.sort(key=lambda x: (x[0], x[1]))
    cur, mx = 0, 0
    for _, delta in tl:
        cur += delta
        mx = max(mx, cur)
    poly_q = max(0, min(9, mx - 1))

    # duration levels (15th/85th percentile across notes in this bar)
    levels = sorted(_dur_level(d) for _, _, d in notes)
    n = len(levels)
    dl15 = levels[max(0, int(n * 0.15))]
    dl85 = levels[min(n - 1, int(n * 0.85))]

    return {"poly_max": mx, "poly_q": poly_q,
            "min_dur_q": dl15, "max_dur_q": dl85, "n_notes": n}


def compute_track_controls(all_bar_ev_indices, all_events):
    """
    Compute TRACK-level polyphony_q and duration_q the same way the C++
    attribute_control.h does: 15th/85th percentile across ALL bars.
    all_bar_ev_indices: list of bar event index lists (one per bar).
    """
    poly_per_bar, all_dur_levels = [], []
    for bar_evs in all_bar_ev_indices:
        r = compute_bar_controls(bar_evs, all_events)
        if r["n_notes"] > 0:
            poly_per_bar.append(r["poly_q"])
            notes = _pair_note_durations(bar_evs, all_events)
            all_dur_levels.extend(_dur_level(d) for _, _, d in notes)

    if not poly_per_bar:
        return {"min_poly_q": 0, "max_poly_q": 0, "min_dur_q": 0, "max_dur_q": 0}
    poly_per_bar.sort()
    all_dur_levels.sort()
    n_p, n_d = len(poly_per_bar), len(all_dur_levels)
    return {
        "min_poly_q": poly_per_bar[max(0, int(n_p * 0.15))],
        "max_poly_q": poly_per_bar[min(n_p - 1, int(n_p * 0.85))],
        "min_dur_q":  all_dur_levels[max(0, int(n_d * 0.15))] if n_d else 0,
        "max_dur_q":  all_dur_levels[min(n_d - 1, int(n_d * 0.85))] if n_d else 0,
    }


def format_token_sequence(prompt_tokens, enc, generated=None):
    """
    Return a plain-text block listing every token as:
        [idx]  id   TOKEN_NAME = value

    prompt_tokens — the model's input (everything before the generation cut).
    generated     — optional list of ints from RecordTokenSequenceCallback;
                    appended after a ◄━━ GENERATE ━━► separator.
    """
    lines = []
    for i, tok in enumerate(prompt_tokens):
        lines.append(f"  [{i:4d}] {tok:5d}  {enc.pretty(tok)}")
    lines.append("  ◄━━ GENERATE ━━►")
    if generated:
        for j, tok in enumerate(generated):
            lines.append(f"  [{len(prompt_tokens) + j:4d}] {tok:5d}  {enc.pretty(tok)}")
    return "\n".join(lines)


def load_encoder_from_checkpoint(ckpt_path, forced_encoder=None):
    if forced_encoder:
        enc_name = forced_encoder
    elif not ckpt_path or not os.path.exists(ckpt_path) or os.path.isdir(ckpt_path):
        return midigpt.GhostEncoder(), "GHOST_ENCODER"
    else:
        try:
            import torch
            import json
            extra_files = {"metadata.json": ""}
            torch.jit.load(ckpt_path, _extra_files=extra_files)
            metadata = json.loads(extra_files["metadata.json"])
            enc_name = metadata.get("encoder", "GHOST_ENCODER")
        except Exception as e:
            print(f"  Warning: Could not load metadata from {ckpt_path}, defaulting to GhostEncoder. Error: {e}")
            return midigpt.GhostEncoder(), "GHOST_ENCODER"

    if enc_name == "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER" or enc_name == "YELLOW_ENCODER":
        return midigpt.ElVelocityDurationPolyphonyYellowEncoder(), "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER"
    elif enc_name == "EXPRESSIVE_ENCODER":
        return midigpt.ExpressiveEncoder(), "EXPRESSIVE_ENCODER"
    elif enc_name in ["STEINBERG_WPCS_ENCODER", "STEINBERG_W_P_C_S_ENCODER"]:
        return midigpt.SteinbergWPCSEncoder(), "STEINBERG_WPCS_ENCODER"
    elif enc_name == "SPECTER_ENCODER":
        return midigpt.SpecterEncoder(), "SPECTER_ENCODER"
    elif enc_name == "ORACLE_ENCODER":
        return midigpt.OracleEncoder(), "ORACLE_ENCODER"
    else:
        return midigpt.GhostEncoder(), "GHOST_ENCODER"


# ── Main simulation loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time agent visualization — per-step grid + token view"
    )
    parser.add_argument("--midi",   required=True,  help="Input MIDI file (human tracks)")
    parser.add_argument("--ckpt",   default="",     help="Model checkpoint (.pt); skip inference if empty")
    parser.add_argument("--buffer", type=int, default=4,  help="Initial silent bars before agent starts")
    parser.add_argument("--lookahead", type=int, default=1, help="Lookahead distance k")
    parser.add_argument("--num_anticipated_bars", type=int, default=1, help="Bars per step j")
    parser.add_argument("--model_dim",  type=int, default=8,  help="Context window D in bars")
    parser.add_argument("--mask_gap",   action="store_true",  help="Mask agent gap bars")
    parser.add_argument("--no_masking", action="store_true",  help="Disable all future/context masking (model sees everything)")
    parser.add_argument("--adapt_buffer", action="store_true", help="Start generating early")
    parser.add_argument("--no_infer",   action="store_true",  help="Skip inference, show grid only")
    parser.add_argument("--max_steps",  type=int, default=4, help="Stop after N generation steps")
    parser.add_argument("--human_tracks", type=int, default=1, help="Number of human conditioning tracks")
    parser.add_argument("--agent_instrument", type=int, default=1, help="GM instrument number for agent")
    parser.add_argument("--agent_min_dur", type=str, default=None,
                        help="Agent MIN_NOTE_DURATION per step (0-5, comma-sep for schedule). "
                             "0=<32nd 1=32nd 2=16th 3=8th 4=quarter 5=half+.")
    parser.add_argument("--agent_max_dur", type=str, default=None,
                        help="Agent MAX_NOTE_DURATION per step (0-5, comma-sep for schedule).")
    parser.add_argument("--agent_min_poly", type=str, default=None,
                        help="Agent MIN_POLYPHONY per step (1-indexed status, comma-sep). "
                             "1=mono 2=2note 3=3note … 0=no-override.")
    parser.add_argument("--agent_max_poly", type=str, default=None,
                        help="Agent MAX_POLYPHONY per step (1-indexed status, comma-sep).")
    parser.add_argument("--agent_tension", type=str, default=None,
                        help="Agent tension decile per step (1=lowest…10=highest, 0=any). "
                             "Comma-sep for schedule. Only effective with OracleEncoder.")
    parser.add_argument("--auto_controls", action="store_true",
                        help="Auto-derive agent polyphony/duration defaults from conditioning tracks.")
    parser.add_argument("--eval_controls", action="store_true",
                        help="After generation, print requested vs realized control values per bar.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling. If not provided, a random one is chosen.")
    parser.add_argument("--encoder", type=str, default=None,
                        help="Override encoder type (e.g. GHOST_ENCODER, EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER)")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory for the generated MIDI file")
    parser.add_argument("--sf2", type=str, default="/scratch/triana24/models/Arachno.sf2", help="Path to soundfont file (.sf2)")
    parser.add_argument("--fluidsynth", type=str, default="/scratch/triana24/fluidsynth-install/bin/fluidsynth", help="Path to fluidsynth executable")
    parser.add_argument("--no_synth", action="store_true", help="Skip MIDI synthesis")
    parser.add_argument("--log", type=str, default="", help="Path to log file for per-step token sequences (optional)")
    args = parser.parse_args()

    D  = args.model_dim
    B  = args.buffer
    k  = args.lookahead
    j  = args.num_anticipated_bars

    if k + j >= D:
        print(f"ERROR: lookahead({k}) + num_anticipated({j}) must be < model_dim({D}).")
        sys.exit(1)
    if B < 2:
        print("ERROR: buffer must be >= 2.")
        sys.exit(1)

    run_inference = (not args.no_infer) and bool(args.ckpt)

    # ── Control schedules (resolved after MIDI load below) ────────────────────
    _ctrl_schedules_raw = (args.agent_min_poly, args.agent_max_poly,
                           args.agent_min_dur,  args.agent_max_dur)

    # per-step eval tracking
    _control_eval = []  # list of dicts: step, bar, requested, realized

    log_file = None
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        log_file = open(args.log, "w", encoding="utf-8")
        log_file.write(f"step_viz log\nmidi={args.midi}  ckpt={args.ckpt}\n"
                       f"D={args.model_dim}  buffer={args.buffer}  k={args.lookahead}"
                       f"  j={args.num_anticipated_bars}  max_steps={args.max_steps}\n"
                       + "=" * 90 + "\n")

    # ── Load Encoder and Model Metadata ────────────────────────────────────────
    enc, enc_name = load_encoder_from_checkpoint(args.ckpt, args.encoder)
    supports_masking = (enc_name in ["GHOST_ENCODER", "SPECTER_ENCODER", "ORACLE_ENCODER"])
    
    if not supports_masking and not args.no_masking:
        print(f"  {_c(YEL, f'Encoder {enc_name} does not support causal masking. Forcing --no_masking.')}")
        args.no_masking = True

    try:
        piece_raw = json.loads(enc.midi_to_json(args.midi))
    except Exception as e:
        print(f"ERROR loading MIDI: {e}")
        sys.exit(1)

    # Sanitize track types (GhostEncoder only supports 10 and 11)
    for t in piece_raw.get("tracks", []):
        tt = t.get("track_type", 10)
        if tt == 8:   t["track_type"] = 11
        elif tt == 9: t["track_type"] = 10
        elif tt not in (10, 11): t["track_type"] = 10

    num_human = min(args.human_tracks, len(piece_raw["tracks"]))
    song_bars = len(piece_raw["tracks"][0]["bars"])
    pad_bars  = max(0, D - song_bars)
    total_bars = song_bars + pad_bars

    # Pad human tracks
    human_tracks = piece_raw["tracks"][:num_human]
    for t in human_tracks:
        t["bars"].extend([{"ts_numerator": 4, "ts_denominator": 4}
                          for _ in range(pad_bars)])

    # Empty agent track
    agent_bars_data = [{"ts_numerator": 4, "ts_denominator": 4}
                       for _ in range(total_bars)]

    # Preserve all top-level metadata (tempo, tempoChanges, internalTicksPerQuarter, …)
    piece = dict(piece_raw)
    piece["tracks"] = human_tracks + [{
        "track_type":  10,
        "instrument":  args.agent_instrument,
        "bars":        agent_bars_data,
    }]

    agent_idx = num_human

    # ── Resolve control schedules ─────────────────────────────────────────────
    _MAX_STEPS = args.max_steps
    if args.auto_controls and any(v is None for v in _ctrl_schedules_raw):
        _auto = derive_controls_from_piece(piece_raw, num_human)
        print(f"  {_c(CYN, f'Auto-derived agent controls: min_poly={_auto[0]} max_poly={_auto[1]} min_dur={_auto[2]} max_dur={_auto[3]}')}")
        _def_min_poly = str(_ctrl_schedules_raw[0]) if _ctrl_schedules_raw[0] is not None else str(_auto[0])
        _def_max_poly = str(_ctrl_schedules_raw[1]) if _ctrl_schedules_raw[1] is not None else str(_auto[1])
        _def_min_dur  = str(_ctrl_schedules_raw[2]) if _ctrl_schedules_raw[2] is not None else str(_auto[2])
        _def_max_dur  = str(_ctrl_schedules_raw[3]) if _ctrl_schedules_raw[3] is not None else str(_auto[3])
    else:
        # Sensible melodic defaults: min_poly=2(2note), max_poly=4(4note), min_dur=3(8th), max_dur=5(quarter)
        _def_min_poly = str(_ctrl_schedules_raw[0]) if _ctrl_schedules_raw[0] is not None else "2"
        _def_max_poly = str(_ctrl_schedules_raw[1]) if _ctrl_schedules_raw[1] is not None else "4"
        _def_min_dur  = str(_ctrl_schedules_raw[2]) if _ctrl_schedules_raw[2] is not None else "3"
        _def_max_dur  = str(_ctrl_schedules_raw[3]) if _ctrl_schedules_raw[3] is not None else "5"

    _sched_min_poly = parse_schedule(_def_min_poly, _MAX_STEPS)
    _sched_max_poly = parse_schedule(_def_max_poly, _MAX_STEPS)
    _sched_min_dur  = parse_schedule(_def_min_dur,  _MAX_STEPS)
    _sched_max_dur  = parse_schedule(_def_max_dur,  _MAX_STEPS)
    _sched_tension  = parse_schedule(args.agent_tension, _MAX_STEPS)

    print(f"  {_c(CYN, 'Control schedule (1-indexed status values):')}")
    for si in range(min(_MAX_STEPS, 8)):
        print(f"    step {si+1}: min_poly={_sched_min_poly[si]} max_poly={_sched_max_poly[si]} "
              f"min_dur={_sched_min_dur[si]} max_dur={_sched_max_dur[si]} "
              f"tension={_sched_tension[si]}")

    # ── Simulation loop ────────────────────────────────────────────────────────
    playhead      = 0
    step_count    = 0
    gen_steps     = 0
    last_gen_bar  = -1

    SEP = "═" * 90

    while playhead < total_bars and gen_steps < args.max_steps:
        # Determine whether to generate this tick
        should_gen = False
        target_bar = None

        if args.adapt_buffer:
            if playhead + k >= B:
                target_bar = playhead + k
                should_gen = True
        else:
            if playhead >= B:
                target_bar = playhead + k
                should_gen = True

        if target_bar is not None:
            if target_bar < B or target_bar >= total_bars:
                should_gen = False
                target_bar = None

        if not should_gen:
            playhead += 1
            step_count += 1
            continue

        num_anticipation = min(j, total_bars - target_bar)
        step_advance = num_anticipation  # amount to advance playhead

        # In the very first step, force generation from bar 0 up to the original target_bar + j
        # to fill the initial buffer bars that would otherwise remain silent.
        # However, C++ engine asserts bars_per_step < model_dim, so cap it to D-1.
        if gen_steps == 0:
            target_end = min(target_bar + step_advance, total_bars)
            num_anticipation = min(target_end, D - 1)
            target_bar = target_end - num_anticipation

        t_start, window_end = compute_window(target_bar, num_anticipation, D, total_bars)

        gen_steps += 1

        # ── Header ────────────────────────────────────────────────────────────
        early = (window_end > target_bar + num_anticipation - 1)
        phase = _c(RED, "EARLY") if early else _c(GRN, "NORMAL")
        print(f"\n{_c(BOLD, SEP)}")
        print(
            f"  {_c(BOLD, f'STEP {gen_steps}')}  "
            f"playhead={_c(CYN, str(playhead))}  "
            f"target={_c(BOLD+CYN, str(target_bar))}  "
            f"j={j}  "
            f"window=[{_c(BOLD, str(t_start))},{_c(BOLD, str(window_end))}]  "
            f"D={D}  "
            f"phase={phase}"
        )
        if early:
            disc_range = f"[{target_bar+num_anticipation}, {window_end}]"
            print(f"  {_c(RED, f'Early phase: bars {disc_range} on agent track generated but DISCARDED by piece_insert')}")
        print(_c(BOLD, SEP))

        # ── Full-piece grid ────────────────────────────────────────────────────
        print(f"\n{_c(BOLD+WHT, '  FULL PIECE GRID')}  ({total_bars} bars)")
        print(render_full_grid(piece, num_human, playhead, target_bar, num_anticipation,
                               t_start, window_end, args.mask_gap, total_bars))

        # ── Window zoom ────────────────────────────────────────────────────────
        print(f"\n{_c(BOLD+WHT, '  WINDOW ZOOM')}  (model_dim={D}, bars {t_start}–{window_end})")
        print(render_window_zoom(piece, num_human, playhead, target_bar, num_anticipation,
                                 t_start, window_end, args.mask_gap))

        # ── Token view ─────────────────────────────────────────────────────────
        print(f"\n{_c(BOLD+WHT, '  TOKEN VIEW')}  (structural tokens only, human=full window, agent=up to GENERATE)")
        token_view_str = render_token_view(piece, num_human, playhead, target_bar,
                                           t_start, window_end, args.mask_gap)
        print(token_view_str)

        # Build prompt token sequence for logging (always, if log requested)
        _prompt_tokens = None
        if log_file:
            try:
                _prompt_tokens = build_input_tokens(
                    piece, num_human, agent_idx, playhead,
                    target_bar, num_anticipation, t_start, window_end,
                    args.mask_gap, enc,
                    agent_min_dur=args.agent_min_dur,
                    agent_max_dur=args.agent_max_dur,
                    agent_min_poly=args.agent_min_poly,
                    agent_max_poly=args.agent_max_poly,
                    no_masking=args.no_masking,
                )
            except Exception as _e:
                _prompt_tokens = None
                print(f"  {_c(YEL, f'Warning: could not build token sequence for log: {_e}')}")

        if log_file:
            phase_str = "EARLY" if early else "NORMAL"
            log_file.write(
                f"\nSTEP {gen_steps}  playhead={playhead}  target={target_bar}"
                f"  j={j}  window=[{t_start},{window_end}]  D={D}  phase={phase_str}\n"
            )
            if early:
                log_file.write(
                    f"  Early phase: bars [{target_bar+num_anticipation},{window_end}]"
                    " on agent track generated but DISCARDED by piece_insert\n"
                )
            log_file.write("TOKEN VIEW:\n")
            log_file.write(_strip_ansi(token_view_str) + "\n")

        # ── Inference ─────────────────────────────────────────────────────────
        _rec_cb = None
        if run_inference:
            # Per-step control values from schedule (1-indexed: 0=no override)
            _step_idx = gen_steps - 1  # gen_steps already incremented above
            _min_poly = _sched_min_poly[_step_idx] or 0
            _max_poly = _sched_max_poly[_step_idx] or 0
            _min_dur  = _sched_min_dur[_step_idx]  or 0
            _max_dur  = _sched_max_dur[_step_idx]  or 0
            _tension  = _sched_tension[_step_idx]  or 0
            print(f"  {_c(CYN, f'Controls this step: min_poly={_min_poly} max_poly={_max_poly} min_dur={_min_dur} max_dur={_max_dur} tension={_tension}')}")

            # Build status
            human_status_bars = []
            for i in range(num_human):
                human_status_bars.append(
                    [{"future": False if args.no_masking else b >= playhead}
                     for b in range(total_bars)]
                )

            agent_status_bars = []
            sel = [False] * total_bars
            for b in range(target_bar, min(target_bar + num_anticipation, total_bars)):
                sel[b] = True
            for b in range(total_bars):
                if args.no_masking:
                    is_future = False
                else:
                    is_future = (b >= target_bar + num_anticipation) or (args.mask_gap and playhead <= b < target_bar)

                if is_future:
                    agent_status_bars.append({"future": True})
                else:
                    bar_entry = {"future": False}
                    if _tension > 0:
                        bar_entry["tension"] = _tension
                    agent_status_bars.append(bar_entry)

            status = {"tracks": []}
            for i in range(num_human):
                status["tracks"].append({
                    "track_id": i,
                    "track_type": piece["tracks"][i].get("track_type", 10),
                    "selected_bars": [False] * total_bars,
                    "suffix_autoregressive": False,
                    "polyphony_hard_limit": 10,
                    "bars": human_status_bars[i],
                })

            agent_status_track = {
                "track_id": agent_idx,
                "track_type": 10,
                "selected_bars": sel,
                "suffix_autoregressive": True,
                "polyphony_hard_limit": 10,
                "instrument": _GM_MAPPING.get(args.agent_instrument % 128, "acoustic_grand_piano"),
                "bars": agent_status_bars,
                "min_note_duration_q": _min_dur,
                "max_note_duration_q": _max_dur,
                "min_polyphony_q":     _min_poly,
                "max_polyphony_q":     _max_poly,
            }
            status["tracks"].append(agent_status_track)

            seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

            params = {
                "model_dim":          D,
                "temperature":        0.8,
                "batch_size":         1,
                "ckpt":               args.ckpt,
                "bars_per_step":      num_anticipation,
                "tracks_per_step":    1,
                "percentage":         100,
                "polyphony_hard_limit": 10,
                "sampling_seed":      seed,
            }

            #print(status, params)

            try:
                _cb_mgr = None
                if log_file:
                    _rec_cb = midigpt.RecordTokenSequenceCallback()
                    _cb_mgr = midigpt.CallbackManager()
                    _cb_mgr.add_callback(_rec_cb)

                res_str, attempts = midigpt.sample_multi_step(
                    json.dumps(piece),
                    json.dumps(status),
                    json.dumps(params),
                    5,
                    _cb_mgr,
                )
                res       = json.loads(res_str)
                # Re-run preprocess_piece to populate internalFeatures (e.g. tension)
                # for the newly generated agent bars.
                if hasattr(enc, "update_internal_features"):
                    try:
                        res = json.loads(enc.update_internal_features(res_str))
                    except Exception:
                        pass
                res_agent = res["tracks"][agent_idx]

                print(f"\n{_c(BOLD+WHT, '  INFERENCE RESULT')}  (attempts={attempts})")
                total_generated = 0
                for b_off in range(num_anticipation):
                    b_global = target_bar + b_off
                    if b_global >= total_bars:
                        break
                    nev = len(res_agent["bars"][b_global].get("events", []))
                    total_generated += nev
                    status_str = _c(BOLD+CYN, f"{nev} events") if nev > 0 else _c(GRY, "silent")
                    print(f"    bar {b_global}: {status_str}")

                print(f"    {_c(BOLD, 'total generated events:')} {total_generated}")

                # ── Control adherence eval ─────────────────────────────────
                if args.eval_controls:
                    all_ev = res.get("events", [])
                    for b_off in range(num_anticipation):
                        b_global = target_bar + b_off
                        if b_global >= total_bars:
                            break
                        bar_ev_indices = res_agent["bars"][b_global].get("events", [])
                        realized = compute_bar_controls(bar_ev_indices, all_ev)
                        # requested values (0-indexed feature level = status - 1)
                        req_min_poly = (_min_poly - 1) if _min_poly > 0 else None
                        req_max_poly = (_max_poly - 1) if _max_poly > 0 else None
                        req_min_dur  = (_min_dur  - 1) if _min_dur  > 0 else None
                        req_max_dur  = (_max_dur  - 1) if _max_dur  > 0 else None
                        req_tension  = (_tension  - 1) if _tension  > 0 else None
                        # internalFeatures is per-bar; list of BarFeatures dicts
                        _bar_feats = res_agent["bars"][b_global].get("internalFeatures", [])
                        act_tension = _bar_feats[0].get("tension") if _bar_feats else None
                        tension_str = (f"  tension_req={req_tension} tension_act={act_tension}"
                                       if req_tension is not None or act_tension is not None else "")
                        def _err(req, act):
                            return f"{abs(req-act):.0f}" if req is not None else "N/A"
                        print(f"    {_c(YEL,'EVAL')} bar {b_global}:"
                              f"  poly_req=[{req_min_poly},{req_max_poly}] poly_act_q={realized['poly_q']}(max={realized['poly_max']})"
                              f"  dur_req=[{req_min_dur},{req_max_dur}]"
                              f"  dur_act=[{realized['min_dur_q']},{realized['max_dur_q']}]"
                              f"  notes={realized['n_notes']}{tension_str}")
                        _control_eval.append({
                            "step": gen_steps, "bar": b_global,
                            "req_min_poly": req_min_poly, "req_max_poly": req_max_poly,
                            "act_poly_q": realized["poly_q"],
                            "req_min_dur": req_min_dur,   "req_max_dur": req_max_dur,
                            "act_min_dur": realized["min_dur_q"], "act_max_dur": realized["max_dur_q"],
                            "n_notes": realized["n_notes"],
                            "req_tension": req_tension, "act_tension": act_tension,
                        })

                if log_file:
                    log_file.write(f"INFERENCE (attempts={attempts}):\n")
                    for b_off in range(num_anticipation):
                        b_global = target_bar + b_off
                        if b_global >= total_bars:
                            break
                        nev = len(res_agent["bars"][b_global].get("events", []))
                        log_file.write(f"  bar {b_global}: {nev} events\n")
                    log_file.write(f"  total generated events: {total_generated}\n")
                    if _prompt_tokens is not None:
                        generated_toks = _rec_cb.tokens if _rec_cb else None
                        log_file.write(
                            f"TOKEN SEQUENCE ({len(_prompt_tokens)} prompt tokens"
                            + (f" + {len(generated_toks)} generated" if generated_toks else "")
                            + "):\n"
                        )
                        log_file.write(
                            format_token_sequence(_prompt_tokens, enc, generated_toks) + "\n"
                        )

                # Replace piece with the full result — preserves internalBeatLength
                # and all bar metadata needed for correct json_to_midi output.
                piece = res
                last_gen_bar = max(last_gen_bar,
                                   min(target_bar + num_anticipation - 1, total_bars - 1))

            except Exception as e:
                print(f"  {_c(RED, f'inference error: {e}')}")
                if log_file:
                    log_file.write(f"INFERENCE ERROR: {e}\n")
                    if _prompt_tokens is not None:
                        log_file.write(f"TOKEN SEQUENCE ({len(_prompt_tokens)} prompt tokens):\n")
                        log_file.write(format_token_sequence(_prompt_tokens, enc) + "\n")

        elif log_file and _prompt_tokens is not None:
            # no_infer mode — still log the prompt tokens
            log_file.write(f"TOKEN SEQUENCE ({len(_prompt_tokens)} prompt tokens, no inference):\n")
            log_file.write(format_token_sequence(_prompt_tokens, enc) + "\n")

        # ── Legend (once, after first step) ───────────────────────────────────
        if gen_steps == 1:
            print(f"\n{_c(BOLD, '  LEGEND:')}")
            print(f"  {_legend_line()}")

        if sys.stdin.isatty() and gen_steps < args.max_steps:
            try:
                input(f"\n{_c(GRY, '  [press ENTER for next step]')} ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
        playhead += step_advance

    print(f"\n{_c(BOLD+GRN, 'Simulation complete.')}  {gen_steps} generation steps.")

    # ── Control adherence summary ──────────────────────────────────────────────
    if args.eval_controls and _control_eval:
        print(f"\n{_c(BOLD+YEL, '  CONTROL ADHERENCE SUMMARY')}")
        print(f"  {'bar':>4}  {'req_poly':>9}  {'act_poly':>9}  {'poly_err':>9}  "
              f"{'req_dur':>9}  {'act_dur':>9}  {'dur_err':>8}  {'ten_req':>7}  {'ten_act':>7}  notes")
        print(f"  {'-'*90}")
        poly_errs, dur_errs, ten_errs = [], [], []
        for r in _control_eval:
            ap = r["act_poly_q"]
            rp_lo, rp_hi = r["req_min_poly"], r["req_max_poly"]
            rp_str = f"[{rp_lo},{rp_hi}]" if rp_lo is not None else "   N/A"
            if rp_lo is not None and rp_hi is not None:
                p_err = max(0, rp_lo - ap, ap - rp_hi)
                poly_errs.append(p_err)
            else:
                p_err = None

            ad_lo, ad_hi = r["act_min_dur"], r["act_max_dur"]
            rd_lo, rd_hi = r["req_min_dur"], r["req_max_dur"]
            rd_str = f"[{rd_lo},{rd_hi}]" if rd_lo is not None else "   N/A"
            ad_str = f"[{ad_lo},{ad_hi}]"
            if rd_lo is not None and rd_hi is not None:
                d_err = max(0, rd_lo - ad_hi, ad_lo - rd_hi)
                dur_errs.append(d_err)
            else:
                d_err = None

            rt = r.get("req_tension"); at = r.get("act_tension")
            rt_str = str(rt) if rt is not None else "N/A"
            at_str = str(at) if at is not None else "N/A"
            if rt is not None and at is not None:
                t_err = abs(rt - at)
                ten_errs.append(t_err)
            else:
                t_err = None

            print(f"  {r['bar']:>4}  {rp_str:>9}  {ap:>9}  "
                  f"{'N/A' if p_err is None else p_err:>9}  "
                  f"{rd_str:>9}  {ad_str:>9}  "
                  f"{'N/A' if d_err is None else d_err:>8}  "
                  f"{rt_str:>7}  {at_str:>7}  {r['n_notes']}")
        if poly_errs:
            print(f"\n  Avg poly error: {sum(poly_errs)/len(poly_errs):.2f}  "
                  f"Avg dur error: {sum(dur_errs)/len(dur_errs) if dur_errs else 'N/A':.2f}  "
                  f"Avg tension error: {sum(ten_errs)/len(ten_errs) if ten_errs else 'N/A'}")

        # Track-level summary: 15th/85th percentile across ALL generated bars
        # (matches how C++ attribute_control.h computes the control tokens)
        try:
            agent_idx = next(
                i for i, t in enumerate(piece["tracks"])
                if t.get("role") == "agent" or i == len(piece["tracks"]) - 1
            )
            all_ev_global = piece.get("events", [])
            agent_bar_evs = [
                piece["tracks"][agent_idx]["bars"][b].get("events", [])
                for b in range(len(piece["tracks"][agent_idx].get("bars", [])))
            ]
            tc = compute_track_controls(agent_bar_evs, all_ev_global)
            req_ctrl = _control_eval[0] if _control_eval else {}
            rp_lo = req_ctrl.get("req_min_poly"); rp_hi = req_ctrl.get("req_max_poly")
            rd_lo = req_ctrl.get("req_min_dur");  rd_hi = req_ctrl.get("req_max_dur")
            print(f"\n  {_c(BOLD+YEL,'TRACK-LEVEL (15th/85th pctile across all bars):')}")
            print(f"    poly:  req=[{rp_lo},{rp_hi}]  realized=[{tc['min_poly_q']},{tc['max_poly_q']}]"
                  f"  err=[{max(0,rp_lo-tc['max_poly_q']) if rp_lo is not None else 'N/A'},{max(0,tc['min_poly_q']-rp_hi) if rp_hi is not None else 'N/A'}]")
            print(f"    dur:   req=[{rd_lo},{rd_hi}]  realized=[{tc['min_dur_q']},{tc['max_dur_q']}]"
                  f"  err=[{max(0,rd_lo-tc['max_dur_q']) if rd_lo is not None else 'N/A'},{max(0,tc['min_dur_q']-rd_hi) if rd_hi is not None else 'N/A'}]")
        except Exception:
            pass

    if log_file:
        log_file.write(f"\n{'='*90}\nSimulation complete. {gen_steps} generation steps.\n")
        log_file.close()
        print(f"  {_c(GRY, f'Token sequence log saved to: {args.log}')}")

    # ── Save outcome ──────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    basename = os.path.basename(args.midi)
    if basename.endswith(".mid") or basename.endswith(".midi"):
        name_no_ext = os.path.splitext(basename)[0]
    else:
        name_no_ext = basename
    
    output_path = os.path.join(args.outdir, f"{name_no_ext}_gen.mid")
    
    try:
        # Trim to last generated bar (inclusive) so output isn't padded with
        # the full-length source MIDI when only a few bars were generated.
        if last_gen_bar >= 0:
            n_bars = last_gen_bar + 1
            trimmed = dict(piece)
            trimmed["tracks"] = []
            for t in piece["tracks"]:
                tc = dict(t)
                tc["bars"] = t["bars"][:n_bars]
                trimmed["tracks"].append(tc)
            piece = trimmed
            print(f"  {_c(GRY, f'Trimming to {n_bars} bars (last generated: bar {last_gen_bar})')}")

        json_str = json.dumps(piece)
        enc.json_to_midi(json_str, output_path)
        print(f"  {_c(BOLD+CYN, 'Saved final MIDI to:')} {output_path}")

        # ── Synthesis ─────────────────────────────────────────────────────────
        if not args.no_synth and os.path.exists(args.fluidsynth) and os.path.exists(args.sf2):
            audio_path = output_path.replace(".mid", ".wav")
            print(f"  {_c(BOLD+CYN, 'Synthesizing audio...')} (using {os.path.basename(args.sf2)})")
            
            # fluidsynth -ni -F [output] -r [rate] [sf2] [midi]
            env = os.environ.copy()
            lib_dir = os.path.dirname(args.fluidsynth).replace("/bin", "/lib64")
            if os.path.exists(lib_dir):
                if "LD_LIBRARY_PATH" in env:
                    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env['LD_LIBRARY_PATH']}"
                else:
                    env["LD_LIBRARY_PATH"] = lib_dir

            cmd = [
                args.fluidsynth,
                "-ni",
                "-F", audio_path,
                "-r", "44100",
                args.sf2,
                output_path,
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  {_c(BOLD+CYN, 'Saved synthesized audio to:')} {audio_path}")
            else:
                print(f"  {_c(RED, f'Synthesis failed (exit {result.returncode}):')} {result.stderr}")
        elif not args.no_synth:
            if not os.path.exists(args.fluidsynth):
                print(f"  {_c(YEL, 'Warning: fluidsynth not found at')} {args.fluidsynth} — skipping synthesis.")
            if not os.path.exists(args.sf2):
                print(f"  {_c(YEL, 'Warning: soundfont not found at')} {args.sf2} — skipping synthesis.")

    except Exception as e:
        print(f"  {_c(RED, f'Error saving MIDI/Audio: {e}')}")


if __name__ == "__main__":
    main()
