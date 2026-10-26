import contextlib
import copy
import json
import logging
import math
import random
import time  # Moved from _sample_step
from dataclasses import replace as replace_cfg
from pathlib import Path

from tqdm import tqdm

import midigpt._core as _core
from midigpt.inference import context_fit, planner
from midigpt._converters import from_cpp, to_cpp
from midigpt._core import GenerationStep  # Import GenerationStep directly from _core
from midigpt._types import Bar, Score
from midigpt.inference.config import (
    GenerationRequest,
    InferenceConfig,
    pitch_shape_weight,
    resolve_pitch_allow_set,
    resolve_rhythm_schedule,
    rhythm_grid_weight,
)

log = logging.getLogger(__name__)


class TokenLogger:
    """Collects per-token generation stats and bar summaries.

    Attach to a SamplingSession before calling .run():
        session = engine.session(score, request)
        session.token_logger = TokenLogger("debug_tokens.jsonl")
        result = session.run()

    The JSONL file contains one JSON object per line:
        {"rec": "token", "pos": int, "tt": str, "tv": int,
         "H_bits": float, "n_legal": int, "n_legal_tt": int,
         "legal_tt": [str,...], "top5": [{"tt":str,"tv":int,"p":float},...]}
        {"rec": "bar", "track": int, "bar": int, "generated": bool,
         "n_notes": int, "onset_spq": [int,...], "pitches": [int,...],
         "unique_onsets": int, "pct_at_0": float}
        {"rec": "step_summary", ...}
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self._path.open("w")
        self._step = 0
        self._token_types_seen: dict[str, int] = {}
        self._collapse_count = 0

    def log_token(self, pos: int, token_id: int, tname: str, tval: int,
                  raw_probs_cpu, mask_cpu, vocab):
        """Called once per generated token, before state.advance().

        raw_probs_cpu: softmax(masked_logits, T=1) — model belief after grammar
                       mask only, before temperature/top-k/p shaping.
        mask_cpu:      boolean grammar mask (True = legal).
        """
        import torch
        p = raw_probs_cpu

        # Entropy (bits) of the model's legal distribution at T=1
        nonzero = p[p > 0]
        H = -float((nonzero * nonzero.log2()).sum().item()) if len(nonzero) else 0.0

        # Legal token types from grammar mask
        legal_tt: list[str] = []
        seen_tt: set[str] = set()
        for i, ok in enumerate(mask_cpu.tolist()):
            if ok:
                try:
                    tt = vocab.get_type(i).name
                    if tt not in seen_tt:
                        legal_tt.append(tt)
                        seen_tt.add(tt)
                except Exception:
                    pass

        # Top-5 by raw (T=1) probability — what the model actually prefers
        top_idx = torch.topk(p, min(5, int((p > 0).sum().item()))).indices.tolist()
        top5 = []
        for i in top_idx:
            try:
                tt2 = vocab.get_type(i).name
                _, tv2 = vocab.decode(i)
            except Exception:
                tt2, tv2 = "?", -1
            top5.append({"tt": tt2, "tv": tv2, "p": round(float(p[i].item()), 5)})

        self._token_types_seen[tname] = self._token_types_seen.get(tname, 0) + 1

        rec = {
            "rec": "token",
            "step": self._step,
            "pos": pos,
            "tid": token_id,            # raw vocab index — needed for cross-scoring replay
            "tt": tname,
            "tv": tval,
            "H_bits": round(H, 4),      # model uncertainty after grammar, at T=1
            "n_legal": int(mask_cpu.sum().item()),
            "n_legal_tt": len(legal_tt),
            "legal_tt": legal_tt,        # which token types grammar allows here
            "top5": top5,               # model's top-5 choices (T=1, after mask)
        }
        self._f.write(json.dumps(rec) + "\n")

    def log_grammar_collapse(self, pos: int, n_legal: int):
        self._collapse_count += 1
        self._f.write(json.dumps({
            "rec": "grammar_collapse", "step": self._step,
            "pos": pos, "n_legal": n_legal
        }) + "\n")

    def log_bar_summaries(self, result: Score, request, gap_bars: list[int], tpb_spq: int):
        """Log per-(track,bar) note stats from a decoded result Score."""
        gen_bars = set(gap_bars)
        for ti, track in enumerate(result.tracks):
            # Find what bars this track has
            all_bars: dict[int, list] = {}
            for note in track.bars:  # Score has .bars list of Bar objects
                pass
            # Use bar.notes directly
            for bi, bar in enumerate(track.bars):
                onsets = sorted(set(n.onset_ticks for n in bar.notes))
                n = len(bar.notes)
                n_at_0 = sum(1 for note in bar.notes if note.onset_ticks == 0)
                pitches = sorted(set(note.pitch for note in bar.notes))
                rec = {
                    "rec": "bar",
                    "step": self._step,
                    "track": ti,
                    "bar": bi,
                    "generated": bi in gen_bars,
                    "n_notes": n,
                    "onset_spq": onsets[:16],
                    "unique_onsets": len(onsets),
                    "pct_at_0": round(100 * n_at_0 / n, 1) if n else 0.0,
                    "pitches": pitches[:16],
                }
                self._f.write(json.dumps(rec) + "\n")

    def log_context(self, context_tokens: list[int], vocab):
        """Decode and log the full context token sequence as seen by the model.

        Walks the token IDs and reconstructs which track/bar each belongs to,
        emitting one JSON record per bar. This lets you compare the model's
        view of the context against the source MIDI.
        """
        track_idx = -1
        bar_idx = -1
        bar_tokens: list[dict] = []

        def flush_bar():
            if bar_tokens:
                self._f.write(json.dumps({
                    "rec": "ctx_bar",
                    "step": self._step,
                    "track": track_idx,
                    "bar": bar_idx,
                    "tokens": bar_tokens,
                }) + "\n")

        for tid in context_tokens:
            try:
                tt = vocab.get_type(tid).name
                _, tv = vocab.decode(tid)
            except Exception:
                tt, tv = "?", tid

            if tt == "Track":
                flush_bar()
                bar_tokens = []
                track_idx += 1
                bar_idx = -1
            elif tt == "Bar":
                flush_bar()
                bar_tokens = []
                bar_idx += 1
            else:
                bar_tokens.append({"tt": tt, "tv": tv})

        flush_bar()
        self._f.flush()

    def log_step_summary(self, context_len: int, max_gen_tokens: int, n_gen_tracks: int,
                         request=None, use_velocity: int = -1):
        total = sum(self._token_types_seen.values())
        # Record exact track structure so cross-scoring can replay with matching grammar
        gen_track_ids: list[int] = []
        ctx_track_ids: list[int] = []
        if request is not None:
            for tp in request.tracks:
                if getattr(tp, "ignore", False):
                    continue
                if tp.bars:
                    gen_track_ids.append(tp.id)
                else:
                    ctx_track_ids.append(tp.id)
        rec = {
            "rec": "step_summary",
            "step": self._step,
            "context_len": context_len,
            "max_gen_tokens": max_gen_tokens,
            "n_gen_tracks": n_gen_tracks,
            "gen_track_ids": gen_track_ids,   # which track IDs were generated
            "ctx_track_ids": ctx_track_ids,   # which track IDs were context (bars=[])
            "use_velocity": use_velocity,     # 0/1 from context encoding; -1=unknown
            "tokens_generated": total,
            "grammar_collapses": self._collapse_count,
            "token_type_counts": dict(sorted(
                self._token_types_seen.items(), key=lambda x: -x[1])),
        }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        self._step += 1
        self._token_types_seen = {}
        self._collapse_count = 0

    def close(self):
        self._f.flush()
        self._f.close()

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass


class _ContextOverflow(Exception):
    """Raised when an encoded step exceeds the model's context length."""

    def __init__(self, ctx_len: int, model_max: int):
        super().__init__(f"context_len={ctx_len} >= model max {model_max}")
        self.ctx_len = ctx_len
        self.model_max = model_max


class _BatchNotEligible(Exception):
    """Raised internally when cfg.batch_attempts can't be honored for this
    step (currently: any hidden span present — batched span-masking isn't
    implemented). Caught by _run_step, which falls back to the sequential
    per-attempt path."""


# Tokens between progress_sink calls.
PROGRESS_EVERY = 32
# Least time between two decodes for note_sink: each decodes the whole step so far, so decoding after every note
# cost a share of the generation that grew with its length. The step's end sends whatever is left.
STREAM_EVERY_S = 0.25   # the Live device forwards progress every 200 ms


class GenerationCancelled(Exception):
    """Raised when SamplingSession.cancel_event is observed set mid-decode.

    Deliberately NOT a RuntimeError — _run_step's retry loop only catches
    RuntimeError (for rejectable "constraint graph error" attempts), so this
    propagates straight through every retry/candidate loop uncaught and out
    of run()/run_variations() to the caller. Retrying after a deliberate
    cancel would be wasted work and would delay honoring the cancellation.

    Carries whatever was generated before the cancellation, decoded through
    the exact same path a natural completion would use, so callers get a
    usable partial result instead of nothing:
      - from _sample_step: `partial` is a single Score.
      - from _sample_step_batch: `partial` is the same list of
        (Score|None, gen_count, error, truncated) tuples _sample_step_batch
        normally returns, one per row.
    """

    def __init__(self, partial, gen_count: int):
        super().__init__("generation cancelled")
        self.partial = partial
        self.gen_count = gen_count


class _KVRunner:
    """Owns the KV cache across a single sampling step.

    Hides past_kv bookkeeping from the main sampling loop. Tracks whether the
    next forward() is the prefill (no cache yet) or a decode step (cache
    populated). Delegates cache surgery to the model's ModelBase methods so the
    session code stays architecture-agnostic.
    """

    def __init__(self, model, initial_kv):
        self._model = model
        self._initial_kv = initial_kv  # cached empty-kv from engine.warmup()
        self._past_kv = None

    @property
    def is_prefill(self) -> bool:
        return self._past_kv is None

    def forward(self, ctx_t, key_mask=None, position_ids=None):
        """Run one model call. Returns logits tensor (full output[0]).

        On the first call past_kv is sourced from the cached empty-kv (avoids
        a redundant make_empty_kv() per step). On subsequent calls the cache
        from the previous step is reused.
        """
        # The C++ encoder always produces CPU tensors; move to model device.
        if self._initial_kv:
            dev = self._initial_kv[0][0].device
            ctx_t = ctx_t.to(dev)
            if key_mask is not None:
                key_mask = key_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        kwargs = {}
        if key_mask is not None:
            kwargs["key_mask"] = key_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        pkv = self._past_kv if self._past_kv is not None else self._initial_kv
        try:
            outputs = (
                self._model(ctx_t, pkv, **kwargs)
                if pkv is not None
                else self._model(ctx_t, **kwargs)
            )
        except Exception:
            # TorchScript callables with strict signatures may reject kwargs
            # or kv on certain code paths — fall back to positional-only.
            outputs = self._model(ctx_t)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self._past_kv = outputs[1] if len(outputs) > 1 else None
        return outputs[0]

    def null_positions(self, spans):
        """Neutralize KV at the given (s, e) spans (attention_approx)."""
        if self._past_kv is None or not spans:
            return
        self._model.kv_null_positions(self._past_kv, spans)


class _RhythmForcer:
    """Per-row state machine enforcing controls["rhythm_mask"]["positions"]
    (hard exact rhythm + per-position polyphony) on one track's generated
    bars. One instance per generated row (sequential: one; batched: one per
    row — see _sample_step_batch) so a row that dies early never corrupts
    another row's progress.

    Bar grammar observed empirically on real checkpoints (yellow_medium):
    every bar opens with an implicit note at tick 0 — no token for leading
    silence — then alternates between "more notes at this tick" (another
    [VelocityLevel] NoteOnset NoteDuration group, no TimeAbsolutePos in
    between — a chord) and "move to a new tick" (a TimeAbsolutePos token),
    until BarEnd. This class narrows mask_buf to walk that exact alternation
    against a fixed schedule: `apply()` restricts the CURRENT step's legal
    tokens, `advance()` updates progress once the actual token is known.

    VelocityLevel is legal in BOTH situations — it can restate the velocity
    for the next chord note OR for the first note after a position move —
    and always commits the grammar to a NoteOnset immediately next (never
    observed followed by anything else). So once a group's target polyphony
    is met, VelocityLevel must be blocked alongside NoteOnset — leaving it
    open would let the model "escape" the forced transition by restating
    velocity, which the grammar then obligates to another onset that's
    already been blocked, deadlocking (n_legal==0).

    Deliberately narrows only — never widens a bit the grammar mask (or any
    other active hard constraint, e.g. density_hard_limit) already turned
    off. A schedule that's incompatible with another hard constraint (e.g.
    requires fewer simultaneous notes than a note_density floor) hits the
    existing n_legal==0 safety net instead of corrupting state — but since
    this forcing is deterministic (not seed-dependent), such a conflict
    fails identically on every retry attempt.
    """

    def __init__(self, schedule: list[tuple[int, int]], pos_range: tuple[int, int],
                 onset_range: tuple[int, int], vel_range: tuple[int, int] | None, barend_id: int):
        self.schedule = schedule  # [(tick, polyphony), ...], sorted, schedule[0][0] == 0
        self.pos_lo, self.pos_hi = pos_range
        self.onset_lo, self.onset_hi = onset_range
        self.vel_range = vel_range  # None if this checkpoint has no VelocityLevel domain
        self.barend_id = barend_id
        self.group_idx = 0
        self.notes_in_group = 0

    def reset(self):
        self.group_idx = 0
        self.notes_in_group = 0

    def apply(self, mask_buf):
        if self.group_idx >= len(self.schedule):
            # Schedule exhausted for this bar — no more notes, no more moves.
            mask_buf[self.onset_lo:self.onset_hi] = False
            mask_buf[self.pos_lo:self.pos_hi] = False
            if self.vel_range is not None:
                mask_buf[self.vel_range[0]:self.vel_range[1]] = False
            return
        required = self.schedule[self.group_idx][1]
        if self.notes_in_group < required:
            # Still filling this onset point — force another NoteOnset
            # (optionally preceded by a VelocityLevel restate, left alone).
            mask_buf[self.pos_lo:self.pos_hi] = False
            mask_buf[self.barend_id] = False
            return
        # This onset point is full — force a move, never another chord note.
        mask_buf[self.onset_lo:self.onset_hi] = False
        if self.vel_range is not None:
            mask_buf[self.vel_range[0]:self.vel_range[1]] = False
        if self.group_idx + 1 < len(self.schedule):
            target_tick = self.schedule[self.group_idx + 1][0]
            target_id = self.pos_lo + target_tick
            # Never force a token the grammar itself hasn't already legalized
            # here — narrow to it only if it's already True.
            still_legal = bool(mask_buf[target_id])
            mask_buf[self.pos_lo:self.pos_hi] = False
            mask_buf[self.barend_id] = False
            if still_legal:
                mask_buf[target_id] = True
        else:
            # Last scheduled onset point — only BarEnd should remain.
            mask_buf[self.pos_lo:self.pos_hi] = False

    def advance(self, tname: str):
        # Reset on BarEnd, not Bar: BarEnd unambiguously marks "this bar's
        # schedule is fully consumed, the next token starts a new bar."
        # Bar itself doesn't work as a trigger — the very first Bar token
        # (opening bar_idx=0, which is already where __init__ starts) would
        # incorrectly fire a reset too, and if a bar's opening is inherited
        # from context rather than sampled, its Bar token never appears in
        # this loop at all.
        if tname == "BarEnd":
            self.reset()
        elif tname == "NoteOnset":
            # Count on NoteOnset, not NoteDuration: drum tracks emit no
            # NoteDuration, so a drum note would never be counted and the
            # forcer would keep demanding notes at the same onset. Nothing
            # here forces durations, so counting at the onset is exact for
            # melodic tracks too.
            self.notes_in_group += 1
        elif tname == "TimeAbsolutePos":
            self.group_idx += 1
            self.notes_in_group = 0


class _VariationForcer:
    """Per-row state machine enforcing controls["remix"] — regenerates a
    track's bars as a partial variation of reference content already
    present in the input score (before this call), instead of generating
    fresh. One instance per generated row (see _RhythmForcer for why).

    The onset SCHEDULE (ticks + polyphony per bar) is always forced from
    the reference, exactly like a _RhythmForcer built from that same
    reference — see that class's docstring for the underlying grammar and
    the narrow-only/never-widen safety philosophy, both fully reused here.
    On top of the schedule, each reference note's pitch (and, in "full"
    mode, duration) is independently rolled once per note: kept forced to
    the reference value, or left open for the model to resample freely
    (subject to whatever other masks — e.g. controls["pitch_mask"] — are
    also active). Velocity is always left to free sampling in both modes:
    this checkpoint exposes no raw-velocity -> VelocityLevel-token
    quantizer, so there's no reliable way to compute the token id a
    reference velocity would encode to.
    """

    def __init__(self, mode: str, amount: float, bars_schedule: list,
                 pos_range: tuple[int, int], onset_range: tuple[int, int],
                 dur_range: tuple[int, int] | None, vel_range: tuple[int, int] | None,
                 barend_id: int, count_on_onset: bool = False):
        # Notes are counted on NoteDuration (a note's duration is forced
        # from the same reference note as its pitch), except on drum
        # tracks, which emit no NoteDuration: there each note is counted
        # on its NoteOnset.
        self.count_on = "NoteOnset" if count_on_onset else "NoteDuration"
        self.mode = mode
        self.amount = amount
        # bars_schedule: one entry per generated bar (generation order), each
        # a list of (tick, [(pitch, duration_ticks), ...]) groups sorted by
        # tick; chord notes within a group sorted by pitch (deterministic —
        # forcing order within a chord doesn't matter, each slot just needs
        # a stable reference value to compare against).
        self.bars_schedule = bars_schedule
        self.pos_lo, self.pos_hi = pos_range
        self.onset_lo, self.onset_hi = onset_range
        self.dur_range = dur_range
        self.vel_range = vel_range
        self.barend_id = barend_id
        self.bar_idx = 0
        self.group_idx = 0
        self.note_idx = 0
        self._roll_note()

    def _current_bar(self):
        return self.bars_schedule[self.bar_idx] if self.bar_idx < len(self.bars_schedule) else []

    def _current_group(self):
        bar = self._current_bar()
        return bar[self.group_idx] if self.group_idx < len(bar) else None

    def _roll_note(self):
        # True = force the reference value; False = leave open for resampling.
        keep = random.random() >= self.amount
        self._force_pitch = keep
        self._force_duration = True if self.mode == "pitch" else keep

    def _narrow(self, mask_buf, rng, value):
        if rng is None:
            return
        lo, hi = rng
        target = lo + value
        if lo <= target < hi and bool(mask_buf[target]):
            mask_buf[lo:hi] = False
            mask_buf[target] = True
        # else: reference value out of this checkpoint's domain, or not
        # currently grammar-legal (context differs from when the reference
        # was made) — leave the sub-range as already narrowed and fall back
        # to free sampling for this attribute rather than forcing a token
        # that isn't actually legal here.

    def reset_bar(self):
        self.bar_idx += 1
        self.group_idx = 0
        self.note_idx = 0
        self._roll_note()

    def apply(self, mask_buf):
        group = self._current_group()
        if group is None:
            # Reference bar's schedule exhausted — no more notes, no more moves.
            mask_buf[self.onset_lo:self.onset_hi] = False
            mask_buf[self.pos_lo:self.pos_hi] = False
            if self.vel_range is not None:
                mask_buf[self.vel_range[0]:self.vel_range[1]] = False
            return
        tick, notes = group
        if self.note_idx < len(notes):
            # Still filling this onset point — force another NoteOnset.
            mask_buf[self.pos_lo:self.pos_hi] = False
            mask_buf[self.barend_id] = False
            pitch, duration = notes[self.note_idx]
            if self._force_pitch:
                self._narrow(mask_buf, (self.onset_lo, self.onset_hi), pitch)
            if self._force_duration:
                self._narrow(mask_buf, self.dur_range, duration)
            return
        # Onset point full — force a move, never another chord note (same
        # VelocityLevel-blocking rationale as _RhythmForcer: leaving it open
        # lets the model "escape" via a velocity restate that then commits
        # to another onset we've already blocked, deadlocking).
        mask_buf[self.onset_lo:self.onset_hi] = False
        if self.vel_range is not None:
            mask_buf[self.vel_range[0]:self.vel_range[1]] = False
        bar = self._current_bar()
        if self.group_idx + 1 < len(bar):
            target_tick = bar[self.group_idx + 1][0]
            target_id = self.pos_lo + target_tick
            still_legal = bool(mask_buf[target_id]) if 0 <= target_tick < (self.pos_hi - self.pos_lo) else False
            mask_buf[self.pos_lo:self.pos_hi] = False
            mask_buf[self.barend_id] = False
            if still_legal:
                mask_buf[target_id] = True
        else:
            mask_buf[self.pos_lo:self.pos_hi] = False

    def advance(self, tname: str):
        # See the matching comment on _RhythmForcer.advance — reset on
        # BarEnd, not Bar, for the same off-by-one/context-inherited-bar
        # reasons, which matter MORE here since each bar has a distinct
        # reference (a fixed shared schedule like _RhythmForcer's would mask
        # the bug; per-bar reference values expose it immediately).
        if tname == "BarEnd":
            self.reset_bar()
        elif tname == self.count_on:
            self.note_idx += 1
            self._roll_note()
        elif tname == "TimeAbsolutePos":
            self.group_idx += 1
            self.note_idx = 0


def _sample_infill_bars(request: GenerationRequest, seed: int) -> GenerationRequest:
    """Keep round(p * N) of the N infill (track, bar) targets, at least one,
    chosen uniformly at random with `seed` (p = config.infill_probability).
    The pool spans every infill track, so the count is exact for the request
    as a whole. Dropped bars become context; AR and ignored tracks are left
    untouched."""
    p = getattr(request.config, "infill_probability", 1.0)
    if p >= 1.0:
        return request
    pool = sorted(
        (tp.id, b) for tp in request.tracks if not tp.ignore and not tp.autoregressive for b in set(tp.bars)
    )
    if not pool:
        return request
    k = max(1, int(p * len(pool) + 0.5))
    keep = set(random.Random(seed).sample(pool, k))
    tracks = [
        tp if tp.ignore or tp.autoregressive else replace_cfg(tp, bars=[b for b in tp.bars if (tp.id, b) in keep])
        for tp in request.tracks
    ]
    return replace_cfg(request, tracks=tracks)


class SamplingSession:
    def __init__(self, engine, score: Score, request: GenerationRequest):
        self._engine = engine
        # cfg.seed < 0 means "caller doesn't care about reproducibility" —
        # resolve that to an actual random seed ONCE per session (not
        # per-attempt/per-candidate) so it stays consistent across every
        # attempt/candidate/step in this run, and so the caller can be told
        # what seed was used and replay this exact generation later. Exposed
        # via http/server.py's response ("seed" / "base_seed" field).
        _cfg_seed = getattr(request.config, "seed", -1)
        self.resolved_seed: int = _cfg_seed if _cfg_seed is not None and _cfg_seed >= 0 else random.randint(0, 2**31 - 1)
        # config.infill_probability < 1 keeps a random subset of the infill
        # target bars, picked once per session from the resolved seed so a
        # replayed seed picks the same bars. generated_bars ({track id: bars})
        # is what the session will actually generate, in the request's track
        # ids (exposed in the HTTP response as "generated_bars").
        request = _sample_infill_bars(request, self.resolved_seed)
        self.generated_bars: dict[int, list[int]] = {
            tp.id: sorted(set(tp.bars)) for tp in request.tracks if tp.bars and not tp.ignore
        }
        # Ignored tracks are left out of the prompt entirely: encoded, each one
        # still costs a track header and a MaskBar per bar of every window.
        # The session works on the remaining tracks, renumbered 0..n-1, and
        # _expand() puts the ignored ones back into every result.
        self._full_score = score
        self._full_request = request
        self._set_tracks(request)
        request = self._request
        self.model_forward_time: float = 0.0
        self.encode_time: float = 0.0
        self.decode_time: float = 0.0
        self.enable_profiling: bool = False
        self.gen_count: int = 0
        # Token-budget bookkeeping, set in _sample_step/_sample_step_batch —
        # exposed via http/server.py's "tokens" response block so callers can
        # see how close a request came to the model's context window.
        self.context_len: int = 0
        self.max_context_len: int = 0
        # gen_count when the current step's decode began, and the largest
        # prompt + generated window of any earlier step (see window_tokens)
        self.step_gen_start: int = 0
        self.peak_window: int = 0
        # 1-based retry attempt in progress (sequential path); reported in
        # streamed progress and in the response so clients can show retries.
        self.attempt: int = 1
        # True if generation stopped because gen_count hit max_gen_tokens
        # (context budget exhausted) rather than the grammar/model reaching
        # a natural stopping point (state.complete()) — e.g. all requested
        # bars filled. Exposed via http/server.py's "tokens" response block
        # ("truncated") so callers can tell "this is genuinely done" from
        # "this got cut off, there may have been more".
        self.truncated: bool = False
        # Optional callback fired once per run() with a per-bar snapshot of
        # the first step's prompt state (CONTEXT/MASKED/TO_GENERATE per
        # (track, bar)). Used for client-side comparison/debug; pure read-only.
        self.prompt_state_sink = None
        self._snapshot_sent = False
        # Optional TokenLogger — attach before calling run() to get full token
        # and context diagnostics written to a JSONL file.
        self.token_logger: "TokenLogger | None" = None
        # Optional threading.Event — attach before calling run()/
        # run_variations() to allow mid-decode cancellation. Polled once per
        # token inside _sample_step/_sample_step_batch's decode loop; when
        # set, the loop stops early and GenerationCancelled is raised with
        # whatever was generated so far (see that class's docstring). A
        # threading.Event (not asyncio.Event) because the decode loop runs
        # in a worker thread (http/server.py's run_in_executor), not the
        # event loop.
        self.cancel_event: "threading.Event | None" = None
        # Optional callback — attach before calling run() to receive newly-
        # completed notes as they're generated: called with a list of dicts
        # {"track": int, "bar": int, "pitch": int, "onset_ticks": int,
        # "duration_ticks": int, "velocity": int}, each time NEW notes appear
        # in the decoded partial score compared to the last check. Fires
        # once per completed note (right after its NoteDuration token) in
        # _sample_step only — NOT _sample_step_batch, i.e. not for
        # num_candidates>1 or batch_attempts: streaming N interleaved
        # candidate note-streams over one connection needs each event
        # tagged by candidate index, which is a bigger design question than
        # this first pass covers.
        self.note_sink: "Callable[[list[dict]], None] | None" = None
        # Optional callback, called with no arguments every PROGRESS_EVERY
        # generated tokens in _sample_step (the streamed path), so a client
        # can show throughput and context use even between notes.
        self.progress_sink: "Callable[[], None] | None" = None

        # Context tracks left out so every step fits (context_fit); request track ids. Set by _fit_context.
        self.dropped_context: list[int] = []
        self._split_bps = self._split_tps = 0     # "auto" bars / tracks per step made smaller by _fit_context
        self._fit_context()

    def _set_tracks(self, request):
        """The session works on the request's non-ignored tracks, renumbered 0..n-1 (_kept: their ids)."""
        score = self._full_score
        self._kept = sorted(tp.id for tp in request.tracks if not tp.ignore)
        if len(self._kept) < len(score.tracks):
            py_full = score if isinstance(score, Score) else from_cpp(score)
            score = replace_cfg(py_full, tracks=[py_full.tracks[i] for i in self._kept])
            request = replace_cfg(request, tracks=[
                replace_cfg(tp, id=self._kept.index(tp.id))
                for tp in sorted(request.tracks, key=lambda tp: tp.id) if not tp.ignore
            ])
        self._score = score
        self._request = request

    def _fit_context(self):
        """Make every step of the plan fit the model: prompt + budget_margin x estimated output < the model's
        context (midigpt.inference.context_fit). In order, changing only what the request leaves to the session:
        leave whole context tracks out (context "auto" / "listed"); generate fewer infill tracks per step
        (tracks_per_step "auto"); fewer bars per step (bars_per_step "auto"). The window step-down in run() stays
        the last resort, and keeps whatever was dropped."""
        cfg = self._request.config
        mode = getattr(cfg, "context", "auto")
        max_ctx = self._model_max_context()
        for _ in range(16):                 # each split halves something: a handful of rounds at most
            measured, steps = self._measure_plan()
            if not measured:
                return
            over = lambda dropped: [m[3] - sum(m[4][t] for t in dropped) >= max_ctx for m in measured]  # noqa: E731
            if not any(over(())):
                return
            dropped = []
            if mode != "all":
                for t in self._drop_order(measured, steps):
                    dropped.append(t)
                    if not any(over(dropped)):
                        break
            if dropped and not any(over(dropped)):
                self._drop(dropped)
                return
            if self._split(steps):          # smaller steps, then measure again (no drops kept)
                continue
            if dropped:                     # still over: drop what helps, the step-down does the rest
                self._drop(dropped)
            return

    def _measure_plan(self):
        """Per planned step: (prompt tokens, track of each token, hidden spans, prompt + margin x estimated output,
        tokens per track)."""
        cfg = self._request.config
        steps = self._plan(self._build_selection_mask(), cfg)
        py = self._score if isinstance(self._score, Score) else from_cpp(self._score)
        n, vocab, measured = len(py.tracks), self._engine._tokenizer._vocab, []
        for st in steps:
            gen = set(st.bars_to_generate)
            _, state, _, use_spans, *_ = self._prepare_state(copy.deepcopy(py), st, emit_snapshot=False)
            tokens = list(state.context_tokens())
            agent = next(iter(gen))[0] if st.is_autoregressive else None
            order = [t for t in range(n) if t != agent] + ([agent] if agent is not None else [])
            owners = context_fit.track_of_token(tokens, vocab, order)
            costs = {t: owners.count(t) for t in range(n)}
            per_bar = context_fit.bar_tokens(tokens, vocab, owners)
            need = len(tokens) + cfg.budget_margin * context_fit.estimate_generated(py, st, per_bar)
            measured.append((tokens, owners, list(state.hidden_spans()) if use_spans else [], need, costs))
        return measured, steps

    def _drop_order(self, measured, steps) -> list[int]:
        """Context tracks in the order they go: context_drop as given, then (context "auto") the least attended
        per token on the fullest step, or the role table when that prompt is too long to read."""
        cfg = self._request.config
        py = self._score if isinstance(self._score, Score) else from_cpp(self._score)
        targets = {t for st in steps for t, _ in st.bars_to_generate}
        keep = {self._kept.index(t) for t in cfg.context_keep if t in self._kept}
        candidates = [t for t in range(len(py.tracks)) if t not in targets and t not in keep]
        order = [self._kept.index(t) for t in cfg.context_drop if t in self._kept and self._kept.index(t) in candidates]
        if cfg.context == "auto":
            rest = [t for t in candidates if t not in order]
            tokens, owners, spans, _, costs = max(measured, key=lambda m: m[3])
            att = (context_fit.attention_from_last(self._engine._model, tokens, spans)
                   if len(tokens) < self._model_max_context() else None)
            order += (context_fit.rank_by_attention(att, owners, rest, costs) if att is not None
                      else context_fit.rank_by_roles(py, sorted(targets), rest, costs))
        return order

    def _drop(self, dropped):
        ids = {self._kept[t] for t in dropped}
        self.dropped_context = sorted(set(self.dropped_context) | ids)
        log.info("context: leaving out tracks %s so every step fits", self.dropped_context)
        self._set_tracks(replace_cfg(self._full_request, tracks=[
            replace_cfg(tp, ignore=True, bars=[]) if tp.id in set(self.dropped_context) else tp
            for tp in self._full_request.tracks
        ]))

    def _split(self, steps) -> bool:
        """Smaller steps, for whatever the request left "auto": first fewer infill tracks per step (halved), then
        fewer bars per step (halved, down to one). False when nothing is left to split."""
        cfg = self._request.config
        if cfg.tracks_per_step == 0:
            widest = max((len({t for t, _ in st.bars_to_generate}) for st in steps if not st.is_autoregressive),
                         default=1)
            if widest > 1:
                self._split_tps = max(1, widest // 2)
                return True
        if cfg.bars_per_step == 0:
            # an AR window from bar 0 writes the whole window whatever bars_per_step is: only infill steps and
            # the later AR steps get smaller, and each split must make them smaller
            cur = self._split_bps or max((len({b for _, b in st.bars_to_generate}) for st in steps), default=1)
            if cur > 1:
                self._split_bps = cur // 2
                return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def window_tokens(self) -> int:
        """Tokens in the model's window for the current step: its prompt plus
        what this step generated. gen_count runs over the whole session, so
        context_len + gen_count overstates it from the second step on. The
        batched path counts every row."""
        return self.context_len + self.gen_count - self.step_gen_start

    def peak_window_tokens(self) -> int:
        return max(self.peak_window, self.window_tokens())

    def _expand(self, result):
        """`result` with the ignored tracks of the input score back in place, and the input's tempo
        (tempo isn't tokenized, so a decoded score has the default)."""
        if result is None:
            return result
        full = self._full_score if isinstance(self._full_score, Score) else from_cpp(self._full_score)
        py = result if isinstance(result, Score) else from_cpp(result)
        if len(self._kept) == len(full.tracks):
            return replace_cfg(py, tempo=full.tempo)
        if py.resolution != full.resolution:  # what a context track goes through
            tok = self._engine._tokenizer
            full = tok.normalize_output(tok.normalize_input(copy.deepcopy(full)))
        tracks = [copy.deepcopy(t) for t in full.tracks]
        for i, t_id in enumerate(self._kept):
            tracks[t_id] = py.tracks[i]
        return replace_cfg(py, tracks=tracks, tempo=full.tempo)

    @contextlib.contextmanager
    def _full_track_ids(self):
        """note_sink / prompt_state_sink see the caller's track ids, not the
        renumbered ones (see __init__)."""
        note_sink, state_sink, kept = self.note_sink, self.prompt_state_sink, self._kept
        if note_sink is not None:
            self.note_sink = lambda notes: note_sink([{**n, "track": kept[n["track"]]} for n in notes])
        if state_sink is not None:
            self.prompt_state_sink = lambda snap: state_sink(
                {**snap, "tracks": [{**t, "id": kept[t["id"]]} for t in snap["tracks"]]})
        try:
            yield
        finally:
            self.note_sink, self.prompt_state_sink = note_sink, state_sink

    def run(self) -> Score:
        with self._full_track_ids():
            try:
                return self._expand(self._run())
            except GenerationCancelled as exc:
                raise GenerationCancelled(self._expand(exc.partial), exc.gen_count) from None

    def run_variations(self) -> list:
        """See _run_variations; every candidate score includes the ignored tracks."""
        def expand(out):
            return [{**c, "score": self._expand(c["score"])} for c in out]
        with self._full_track_ids():
            try:
                return expand(self._run_variations())
            except GenerationCancelled as exc:
                raise GenerationCancelled(expand(exc.partial), exc.gen_count) from None

    def _run(self) -> Score:
        self.gen_count = 0
        self.step_gen_start = self.peak_window = 0
        self.attempt = 1
        self.model_forward_time = 0.0
        self.encode_time = 0.0
        self.decode_time = 0.0
        self._snapshot_sent = False

        mask = self._build_selection_mask()
        cfg = self._request.config
        enc_cfg = self._engine._tokenizer._vocab.config()
        import json as _json

        try:
            dims = sorted(
                set(_json.loads(enc_cfg.to_json()).get("num_bars_map") or []),
                reverse=True,
            )
        except Exception:
            dims = []
        # Candidate model_dims to try, largest-first, capped at the requested
        # value. If num_bars_map is unavailable, just try the requested dim.
        candidates = [d for d in dims if d <= cfg.model_dim] or [cfg.model_dim]

        # On a step-down the steps already done are kept: the next smaller
        # model_dim plans only the bars still to generate, with the generated
        # ones as context.
        py_score = self._score if isinstance(self._score, Score) else from_cpp(self._score)
        score = to_cpp(copy.deepcopy(py_score))
        selected = [list(row) for row in mask.selected]
        any_truncated = False
        n_steps = 0
        last_exc = None
        for d in candidates:
            cfg_try = replace_cfg(cfg, model_dim=d) if d != cfg.model_dim else cfg
            enc_cfg.model_dim = d
            mask.selected = selected
            steps = self._order_steps(self._plan(mask, cfg_try), cfg_try)
            # A step that ran out of tokens is missing its last bars (or every
            # attempt came back empty): redo it, and the rest, at the next
            # smaller window, which leaves more of the context for generating.
            # The smallest one keeps the cut-off result (reported as truncated).
            step_down = d != candidates[-1]
            try:
                for step in tqdm(steps):
                    try:
                        result = self._run_step(score, step, cfg_try)
                    except RuntimeError:
                        if step_down and self.truncated:
                            raise _ContextOverflow(self.context_len, self.max_context_len) from None
                        raise
                    if step_down and self.truncated:
                        raise _ContextOverflow(self.context_len, self.max_context_len)
                    score = self._keep_non_targets(score, result, step)
                    any_truncated = any_truncated or self.truncated
                    n_steps += 1
                    for t, b in step.bars_to_generate:
                        selected[t][b] = False
            except _ContextOverflow as exc:
                last_exc = exc
                log.warning(
                    "context overflow at model_dim=%d (ctx=%d, max=%d); stepping down",
                    d,
                    exc.ctx_len,
                    exc.model_max,
                )
                continue
            if n_steps == 0:
                log.warning(
                    "planner produced zero steps at model_dim=%d — nothing was "
                    "generated; returning the input score unchanged",
                    d,
                )
            self.truncated = any_truncated
            # _run_step always hands back a wrapped Score; if no step ran,
            # `score` is still the raw _core.Score from to_cpp() above —
            # normalize either way so callers always get a real Score
            # (with .to_dict()) instead of the native pybind object.
            return score if isinstance(score, Score) else from_cpp(score)
        raise RuntimeError(f"no model_dim in {candidates} fits the model (last: {last_exc})")

    def _plan(self, mask, cfg) -> list:
        """The steps for the bars still selected in `mask`, at cfg.model_dim (midigpt.inference.planner)."""
        n_tracks, n_bars = len(mask.selected), len(mask.selected[0]) if mask.selected else 0
        planned = planner.plan(
            mask.selected, mask.autoregressive, mask.ignore, cfg.model_dim,
            cfg.bars_per_step or getattr(self, "_split_bps", 0),
            cfg.tracks_per_step or getattr(self, "_split_tps", 0),
            cfg.ar_overlap, cfg.min_overlap_bars,
        )
        ignored = {t for t in range(n_tracks) if mask.ignore[t]}
        return [s.to_core(n_tracks, n_bars, ignored) for s in planned]

    def plan_summary(self) -> list[dict]:
        """The plan run() would follow, without generating: one entry per step (window, AR or not, the bars it
        writes per track), at the requested model_dim. Track ids are the request's."""
        mask = self._build_selection_mask()
        steps = self._order_steps(self._plan(mask, self._request.config), self._request.config)
        kept = getattr(self, "_kept", None)
        out = []
        for st in steps:
            gen: dict[str, list[int]] = {}
            for t, b in sorted(st.bars_to_generate):
                gen.setdefault(str(kept[t] if kept else t), []).append(b)
            out.append({"window": [st.start_bar, st.end_bar], "autoregressive": st.is_autoregressive, "generate": gen})
        return out

    def _keep_non_targets(self, before, after, step):
        """An AR step writes its track to the window's end. Bars of that track in the window it was not asked for
        (its history, and bars after a run that ends before the window does) keep what they had."""
        if not step.is_autoregressive:
            return after
        gen = set(step.bars_to_generate)
        ref = before if isinstance(before, Score) else from_cpp(before)
        tok = self._engine._tokenizer
        for t in {t for t, _ in gen}:
            for b in range(step.start_bar, step.end_bar):
                if (t, b) in gen or b >= len(after.tracks[t].bars):
                    continue
                if ref.resolution != after.resolution:   # the first step's input is at the client's resolution
                    ref = tok.normalize_output(tok.normalize_input(copy.deepcopy(ref)))
                after.tracks[t].bars[b] = copy.deepcopy(ref.tracks[t].bars[b])
        return after

    def _order_steps(self, steps: list, cfg) -> list:
        """AR steps in planner order (each continues the last), then the infill
        steps, in random order when cfg.shuffle is set (seeded by the request
        seed when there is one)."""
        if not getattr(cfg, "shuffle", False):
            return steps
        ar = [s for s in steps if s.is_autoregressive]
        infill = [s for s in steps if not s.is_autoregressive]
        seed = self.resolved_seed if self.resolved_seed is not None and self.resolved_seed >= 0 else None
        random.Random(seed).shuffle(infill)
        return ar + infill

    def _run_variations(self) -> list:
        """cfg.num_candidates independent full generations from the SAME
        prompt — one seed per candidate, NO accept/reject filtering (unlike
        the max_attempts retry loop). Meant for "give me a few takes, I'll
        pick one" workflows: e.g. 4 existing tracks, generating a 5th, and
        wanting several differently-seeded options to choose from instead of
        committing to whatever the first generation produces.

        Returns a list of cfg.num_candidates dicts, each
        {"score": Score | None, "seed": int, "gen_count": int, "error": str | None},
        in seed order (index 0 = base seed, index 1 = base seed + 1, ...).

        Fast path: if the request resolves to a single planner step and
        needs no hidden spans (no explicit mask_bars), all N candidates run
        as ONE batched decode (one forward pass per token step across all N
        rows) via _sample_step_batch. Otherwise (multi-step plan, or a
        request that needs span masking) falls back to N sequential
        single-candidate run() calls, each with its own seed — slower, but
        correct for every request shape.
        """
        cfg = self._request.config
        N = cfg.num_candidates
        base_seed = self.resolved_seed
        seeds = [base_seed + i for i in range(N)]

        mask = self._build_selection_mask()
        enc_cfg = self._engine._tokenizer._vocab.config()
        import json as _json

        try:
            dims = sorted(
                set(_json.loads(enc_cfg.to_json()).get("num_bars_map") or []),
                reverse=True,
            )
        except Exception:
            dims = []
        candidates_dims = [d for d in dims if d <= cfg.model_dim] or [cfg.model_dim]
        d = candidates_dims[0]
        cfg_try = replace_cfg(cfg, model_dim=d) if d != cfg.model_dim else cfg
        enc_cfg.model_dim = d
        py_score = self._score if isinstance(self._score, Score) else from_cpp(self._score)
        steps = self._plan(mask, cfg_try)

        if len(steps) == 1:
            self.gen_count = 0
            self.step_gen_start = self.peak_window = 0
            self.model_forward_time = 0.0
            self.encode_time = 0.0
            self.decode_time = 0.0
            self._snapshot_sent = False
            try:
                # _sample_step_batch expects a Python Score — it does its own
                # per-row copy.deepcopy() + to_cpp(), unlike run()'s single-row
                # path which converts to native once upfront.
                raw = self._sample_step_batch(
                    copy.deepcopy(py_score), steps[0], [cfg_try.temperature] * N, seeds
                )
                return [
                    {
                        "score": s, "seed": seeds[i], "gen_count": gc,
                        "error": err, "truncated": trunc,
                    }
                    for i, (s, gc, err, trunc) in enumerate(raw)
                ]
            except _BatchNotEligible as exc:
                log.info(
                    "run_variations: batched path not eligible (%s); falling back "
                    "to %d sequential run() calls",
                    exc, N,
                )
            except _ContextOverflow as exc:
                # the sequential path steps model_dim down per candidate
                log.warning(
                    "run_variations: context overflow at model_dim=%d (ctx=%d, max=%d); "
                    "falling back to %d sequential run() calls",
                    d, exc.ctx_len, exc.model_max, N,
                )
            except GenerationCancelled as exc:
                # exc.partial is _sample_step_batch's own raw tuple list —
                # reshape into the same dict shape a successful return
                # would have, same as the non-cancelled branch above.
                raise GenerationCancelled(
                    [
                        {
                            "score": s, "seed": seeds[i], "gen_count": gc,
                            "error": err, "truncated": trunc,
                        }
                        for i, (s, gc, err, trunc) in enumerate(exc.partial)
                    ],
                    exc.gen_count,
                ) from None

        # Fallback: N independent sequential single-candidate generations.
        # Slower (no batched forward pass), but correct for multi-step plans
        # and span-masked requests, which the batched path doesn't cover.
        original_request = self._request
        original_resolved_seed = self.resolved_seed
        out = []
        # self.run() resets gen_count/model_forward_time/etc. to 0 at its own
        # top, so each candidate's values would clobber the previous one's —
        # accumulate here so the session-level totals (read by
        # http/server.py's aggregate "tokens"/"timing" blocks) cover ALL N
        # candidates, not just whichever ran last.
        total_gen_count = 0
        total_forward = total_encode = total_decode = 0.0
        any_truncated = False
        cancelled = False
        try:
            for i in range(N):
                self._request = replace_cfg(
                    original_request,
                    config=replace_cfg(original_request.config, seed=seeds[i], num_candidates=1),
                )
                # _run_step reads self.resolved_seed, not cfg.seed directly —
                # keep them in sync or every candidate would silently reuse
                # the ORIGINAL (pre-fallback) resolved seed instead of seeds[i].
                self.resolved_seed = seeds[i]
                self._snapshot_sent = False
                try:
                    result = self._run()
                    out.append({
                        "score": result, "seed": seeds[i], "gen_count": self.gen_count,
                        "error": None, "truncated": self.truncated,
                    })
                    any_truncated = any_truncated or self.truncated
                except GenerationCancelled as exc:
                    # Cancellation stops the WHOLE batch here, not just this
                    # candidate — retrying/continuing after a deliberate
                    # cancel would be wasted work and would delay honoring
                    # it. Remaining (not-yet-started) candidates are simply
                    # absent from `out`, same as "never got to them".
                    out.append({
                        "score": exc.partial, "seed": seeds[i], "gen_count": exc.gen_count,
                        "error": None, "truncated": True,
                    })
                    total_gen_count += exc.gen_count
                    cancelled = True
                    break
                except Exception as exc:
                    out.append({
                        "score": None, "seed": seeds[i], "gen_count": 0,
                        "error": str(exc), "truncated": False,
                    })
                total_gen_count += self.gen_count
                total_forward += self.model_forward_time
                total_encode += self.encode_time
                total_decode += self.decode_time
        finally:
            self._request = original_request
            self.resolved_seed = original_resolved_seed
            self.gen_count = total_gen_count
            self.model_forward_time = total_forward
            self.encode_time = total_encode
            self.decode_time = total_decode
            self.truncated = any_truncated or cancelled

        if cancelled:
            raise GenerationCancelled(out, total_gen_count)
        return out

    def _run_step(
        self, score: Score, step: GenerationStep, cfg: InferenceConfig | None = None
    ) -> Score:
        if cfg is None:
            cfg = self._request.config
        # Normalise to _types.Score so helpers can always use bar.notes.
        # On the first step the caller passes a _core.Score; from step 2 on
        # it's already _types.Score (the previous step's candidate).
        if not isinstance(score, Score):
            score = from_cpp(score)
        original_score = score  # baseline for novelty comparison

        errors = []
        base_seed = self.resolved_seed
        last_candidate = None
        import torch
        _device_is_mps = next(self._engine._model.parameters()).device.type == "mps"

        # Batched attempts run in _sample_step_batch, which streams nothing: a streamed generation (note_sink)
        # runs its attempts one after the other instead.
        if getattr(cfg, "batch_attempts", False) and cfg.max_attempts > 1 and self.note_sink is None:
            temperatures = [
                cfg.temperature * (cfg.temperature_escalation**i) for i in range(cfg.max_attempts)
            ]
            seeds = [
                (base_seed + i) if base_seed is not None and base_seed >= 0 else -1
                for i in range(cfg.max_attempts)
            ]
            try:
                batch_results = self._sample_step_batch(score, step, temperatures, seeds)
            except _BatchNotEligible as exc:
                log.info(
                    "batch_attempts requested but not eligible (%s); falling back to sequential",
                    exc,
                )
            else:
                for i, (candidate, gen_count, err, _truncated) in enumerate(batch_results):
                    if err is not None:
                        errors.append(f"sampling crash ({err})")
                        log.warning(
                            "attempt %d/%d crashed (%s) [batched, gen_count=%d]",
                            i + 1, cfg.max_attempts, err, gen_count,
                        )
                        continue
                    last_candidate = candidate
                    silence_failed = self._note_count(candidate, step) <= 0
                    novelty_failed = self._too_similar(original_score, candidate, step, cfg)
                    attempt_errors = []
                    attempt_warnings = []
                    if silence_failed:
                        msg = "silence_check (no notes generated in target bars)"
                        (attempt_errors if cfg.silence_check else attempt_warnings).append(msg)
                    if novelty_failed:
                        msg = self._novelty_message(cfg)
                        (attempt_errors if cfg.novelty_check else attempt_warnings).append(msg)
                    for w in attempt_warnings:
                        log.warning(
                            "attempt %d/%d accepted with warning: %s (check disabled) [batched]",
                            i + 1, cfg.max_attempts, w,
                        )
                    if not attempt_errors:
                        log.info(
                            "attempt %d/%d accepted [batched, gen_count=%d]",
                            i + 1, cfg.max_attempts, gen_count,
                        )
                        return candidate
                    errors.extend(attempt_errors)
                    log.info(
                        "attempt %d/%d rejected (%s) [batched, gen_count=%d]",
                        i + 1, cfg.max_attempts, "; ".join(attempt_errors), gen_count,
                    )

                from collections import Counter
                reason_summary = ", ".join(f"{r}×{c}" for r, c in Counter(errors).most_common())
                raise RuntimeError(
                    f"Max attempts ({cfg.max_attempts}) reached, no acceptable candidate "
                    f"found. Rejection reasons: {reason_summary}. "
                    f"Last candidate had "
                    f"{self._note_count(last_candidate, step) if last_candidate is not None else 0} "
                    f"notes in "
                    f"generated bars. "
                    f"(Hints: vary retries with temperature_escalation > 1, raise "
                    f"max_attempts, disable the failing check, or relax top_p/mask_p "
                    f"for broader sampling.)"
                )

        for i in range(cfg.max_attempts):
            self.attempt = i + 1
            temperature = cfg.temperature * (cfg.temperature_escalation**i)
            # Bump the seed per retry when the user pinned one — otherwise all
            # attempts would resample the same tokens deterministically and
            # max_attempts would be a no-op. seed<0 keeps the RNG free-running.
            attempt_seed = (base_seed + i) if base_seed is not None and base_seed >= 0 else -1
            try:
                candidate = self._sample_step(score, step, temperature, attempt_seed)
            except RuntimeError as exc:
                if "constraint graph error" not in str(exc):
                    raise
                # Grammar deadlock is just another rejectable attempt outcome —
                # a different seed/temperature can land on a different token
                # path that doesn't hit it. Let the retry loop below handle it
                # like any other rejected candidate instead of crashing the
                # whole request on attempt 1.
                errors.append(f"sampling crash ({exc})")
                log.warning(
                    "attempt %d/%d crashed (%s); retrying with temp=%.2f, seed=%s",
                    i + 1,
                    cfg.max_attempts,
                    exc,
                    temperature * cfg.temperature_escalation,
                    attempt_seed,
                )
                if _device_is_mps:
                    torch.mps.empty_cache()
                continue
            last_candidate = candidate
            if _device_is_mps:
                torch.mps.empty_cache()

            # Always evaluate BOTH checks. Whether a failure is an error or just
            # a warning depends on whether the corresponding check is enabled.
            silence_failed = self._note_count(candidate, step) <= 0
            novelty_failed = self._too_similar(original_score, candidate, step, cfg)
            attempt_errors = []
            attempt_warnings = []
            if silence_failed:
                msg = "silence_check (no notes generated in target bars)"
                (attempt_errors if cfg.silence_check else attempt_warnings).append(msg)
            if novelty_failed:
                msg = self._novelty_message(cfg)
                (attempt_errors if cfg.novelty_check else attempt_warnings).append(msg)

            for w in attempt_warnings:
                log.warning(
                    "attempt %d/%d accepted with warning: %s (check disabled)",
                    i + 1,
                    cfg.max_attempts,
                    w,
                )
            if not attempt_errors:
                return candidate

            errors.extend(attempt_errors)
            log.info(
                "attempt %d/%d rejected (%s); retrying with temp=%.2f, seed=%s, "
                "gen_count_so_far=%d",
                i + 1,
                cfg.max_attempts,
                "; ".join(attempt_errors),
                temperature * cfg.temperature_escalation,
                attempt_seed,
                self.gen_count,
            )

        from collections import Counter

        reason_summary = ", ".join(f"{r}×{c}" for r, c in Counter(errors).most_common())  # noqa: RUF001
        raise RuntimeError(
            f"Max attempts ({cfg.max_attempts}) reached, no acceptable candidate "
            f"found. Rejection reasons: {reason_summary}. "
            f"Last candidate had {self._note_count(last_candidate, step)} notes in "
            f"generated bars. "
            f"(Hints: vary retries with temperature_escalation > 1, raise "
            f"max_attempts, disable the failing check, or relax top_p/mask_p "
            f"for broader sampling.)"
        )

    @staticmethod
    def _step_is_drums(score, step) -> bool:
        """True when every track generated in this step is a drum track
        (drum tracks emit no NoteDuration tokens)."""
        step_tracks = {t for (t, _b) in step.bars_to_generate}
        types = {getattr(score.tracks[t], "track_type", "melodic") for t in step_tracks if t < len(score.tracks)}
        return types == {"drum"}

    def _resolve_step_control(self, step, key: str):
        """Resolve a per-track `controls[key]` dict for whichever track(s)
        are active in this step. Steps normally involve exactly one track
        (tracks_per_step=1, the default); if a joint multi-track step has
        tracks with differing values for `key` — INCLUDING one track setting
        it and another leaving it unset — the control is skipped for this
        step (logged) rather than guessed at. There's no per-token track
        attribution available at the point tokens get masked (the sampled
        tokens for every track active in a joint step share one interleaved
        stream), so applying one track's setting to another's tokens would
        be silently wrong rather than merely conservative.
        """
        step_tracks = {t for (t, _b) in step.bars_to_generate}
        values = [tp.controls.get(key) for tp in self._request.tracks if tp.id in step_tracks]
        if not values or all(v is None for v in values):
            return None
        first = values[0]
        if all(v == first for v in values[1:]):
            return first
        log.warning(
            "controls[%r] differs across tracks %s active in this step; "
            "skipping for this step",
            key, sorted(step_tracks),
        )
        return None

    def _build_pitch_mask(self, step):
        """Hard allow-set + soft shape weights for TokenType.NoteOnset,
        derived from whichever track's `controls["pitch_mask"]` applies to
        this step (see _resolve_step_control). Works identically for AR and
        infill steps — it hooks the same post-grammar-mask point already
        used for the delta/velocity hard blocks, before any AR-vs-infill
        branching has happened.

        Returns ((range_start, range_end), allow_bool[domain] | None,
        shape_weights[domain] | None), or (None, None, None) if this
        checkpoint has no NoteOnset domain or no track in this step
        configures pitch_mask.
        """
        import torch

        pitch_mask = self._resolve_step_control(step, "pitch_mask")
        if not pitch_mask:
            return None, None, None
        vocab = self._engine._tokenizer._vocab
        s, e = vocab.range(_core.TokenType.NoteOnset)
        if s < 0 or e <= s:
            return None, None, None
        domain = e - s

        allow_set = resolve_pitch_allow_set(pitch_mask)
        allow_bool = None
        if allow_set is not None:
            allow_bool = torch.zeros(domain, dtype=torch.bool)
            for p in allow_set:
                if 0 <= p < domain:
                    allow_bool[p] = True

        shape = pitch_mask.get("shape")
        shape_weights = None
        if shape:
            shape_weights = torch.tensor(
                [pitch_shape_weight(shape, p) for p in range(domain)], dtype=torch.float32
            )

        return (s, e), allow_bool, shape_weights

    def _build_rhythm_mask(self, step):
        """Resolves controls["rhythm_mask"] for whichever track applies to
        this step (see _resolve_step_control — same single-track-or-agree
        rule as pitch_mask). Works identically for AR and infill steps for
        the same reason _build_pitch_mask does: it hooks the shared
        post-grammar-mask point, before any AR-vs-infill branching.

        Returns (pos_range, schedule | None, grid_weights[domain] | None):
        `schedule` is set for hard "positions" mode (consumed by a fresh
        _RhythmForcer per row); `grid_weights` is set for soft "grid" mode
        (multiplied into probs like the pitch shape weights). At most one of
        the two is non-None — validation.py enforces they're mutually
        exclusive. (None, None, None) if this checkpoint has no
        TimeAbsolutePos domain or no track in this step configures it.
        """
        import torch

        rhythm_mask = self._resolve_step_control(step, "rhythm_mask")
        if not rhythm_mask:
            return None, None, None
        vocab = self._engine._tokenizer._vocab
        s, e = vocab.range(_core.TokenType.TimeAbsolutePos)
        if s < 0 or e <= s:
            return None, None, None
        domain = e - s

        if "positions" in rhythm_mask:
            schedule = resolve_rhythm_schedule(rhythm_mask)
            return (s, e), schedule, None

        grid = rhythm_mask["grid"]
        resolution = int(vocab.config().resolution)
        grid_weights = torch.tensor(
            [rhythm_grid_weight(grid, resolution, t) for t in range(domain)], dtype=torch.float32
        )
        return (s, e), None, grid_weights

    def _build_remix_schedule(self, score, step):
        """Resolves controls["remix"] for whichever track applies to this
        step (see _resolve_step_control). Reference note data is read
        directly from `score` — the bars being varied are still generation
        targets from the model's point of view (masked out of its visible
        context as usual, same as any infill/AR target), but their note
        data is present in the Python object regardless of what the model
        can see, since it's `score` before to_cpp()/SessionState masking.

        Returns (mode, amount, bars_schedule) or (None, None, None).
        `bars_schedule` is one entry per bar in this step's generation
        order, each a list of (tick, [(pitch, duration_ticks), ...]) groups
        sorted by tick (chord notes sorted by pitch — an arbitrary but
        stable order, since forcing only compares values, not positions
        within a chord).
        """
        remix = self._resolve_step_control(step, "remix")
        if not remix:
            return None, None, None
        track_ids = {t for (t, _b) in step.bars_to_generate}
        if len(track_ids) != 1:
            return None, None, None
        tid = next(iter(track_ids))
        bar_ids = sorted(b for (t, b) in step.bars_to_generate if t == tid)
        bars_schedule = []
        for b in bar_ids:
            notes = score.tracks[tid].bars[b].notes
            groups: dict[int, list[tuple[int, int]]] = {}
            for n in notes:
                # NoteDuration's token VALUE is (duration_ticks - 1), not the
                # raw tick count (confirmed against this checkpoint's vocab —
                # e.g. a 240-tick note encodes as NoteDuration value 239).
                groups.setdefault(int(n.onset_ticks), []).append(
                    (int(n.pitch), max(0, int(n.duration_ticks) - 1))
                )
            ordered = [(tick, sorted(groups[tick], key=lambda t: t[0])) for tick in sorted(groups)]
            bars_schedule.append(ordered)
        mode = remix.get("mode", "pitch")
        amount = float(remix.get("amount", 0.0))
        return mode, amount, bars_schedule

    @staticmethod
    def _diff_notes(prev, curr) -> list[dict]:
        """Notes present in `curr` but not in `prev`, per (track, bar).
        Assumes decode order is append-only as generation progresses within
        a bar — true in practice, since each call re-decodes the same
        growing context — so any length increase in a (track, bar)'s note
        list means the newly-appended tail is what's new. `prev` may have
        fewer tracks/bars than `curr` if the score is still being built up;
        treated as "0 prior notes" in that case, not an error.
        """
        if prev is None:
            return []
        new_notes = []
        for t_idx, track in enumerate(curr.tracks):
            prev_track = prev.tracks[t_idx] if t_idx < len(prev.tracks) else None
            for b_idx, bar in enumerate(track.bars):
                prev_bar = (
                    prev_track.bars[b_idx]
                    if prev_track is not None and b_idx < len(prev_track.bars)
                    else None
                )
                prev_count = len(prev_bar.notes) if prev_bar is not None else 0
                if len(bar.notes) > prev_count:
                    for n in bar.notes[prev_count:]:
                        new_notes.append({
                            "track": t_idx,
                            "bar": b_idx,
                            "pitch": n.pitch,
                            "onset_ticks": n.onset_ticks,
                            "duration_ticks": n.duration_ticks,
                            "velocity": n.velocity,
                        })
        return new_notes

    def _prepare_state(self, score, step, emit_snapshot: bool = True):
        """The step's SessionState as _sample_step builds it: analyzer and user attributes on the tracks, the
        context the request implies, piece-level controls. Returns (normalized score, state, mask_mode,
        use_span_masks, encode start time, use_velocity, use_microtiming). Also used to measure prompts before generating (emit_snapshot=False).
        """
        # Three attribute regimes:
        #  - Full AR  (tp.autoregressive, no prefix bars): skip analyzer; user
        #    attributes are forced via AttributeValueConstraint so the model
        #    emits the requested tokens. The encoder emits no attribute tokens
        #    for this track.
        #  - Partial AR (tp.autoregressive, prefix bars exist): treat like
        #    infill — run the analyzer over the prefix, let user attributes
        #    override in the prompt, NO constraint (the prompt already pins
        #    attribute tokens for the prefix bars).
        #  - Infill: run the analyzer; user attributes override in the prompt.
        full_ar_ids: set[int] = set()
        for tp in self._request.tracks:
            if tp.autoregressive and (not tp.bars or min(tp.bars) == 0):
                full_ar_ids.add(tp.id)
        self._full_ar_ids = full_ar_ids  # consumed by _build_constraints
        analyzer = self._engine._analyzer
        if analyzer is not None:
            for t_idx, track in enumerate(score.tracks):
                if t_idx in full_ar_ids:
                    continue
                new_attrs = dict(track.attributes)
                new_attrs.update(analyzer.compute_track_tokens(score, t_idx))
                for b_idx in range(len(track.bars)):
                    for k, v in analyzer.compute_bar_tokens(score, t_idx, b_idx).items():
                        new_attrs[f"bar_{k}_{b_idx}"] = v
                track.attributes = new_attrs

        # User attribute overrides flow into the prompt for infill + partial AR.
        # Full AR uses constraints instead (see _build_constraints).
        for tp in self._request.tracks:
            if tp.id < len(score.tracks) and tp.id not in full_ar_ids:
                score.tracks[tp.id].attributes.update(tp.attributes)

        # Per-bar overrides: write bar_{TokenType}_{bar_idx} keys directly into
        # the prompt, overwriting analyzer-derived values. For infill and
        # partial-AR (suffix bars), the encoder reads these straight from
        # `track.attributes`. For full-AR, per-bar overrides require a C++
        # BarAttributeValueConstraint (Phase 2) — emit a one-time warning and
        # ignore them here so the request still runs.
        for tp in self._request.tracks:
            if tp.id >= len(score.tracks):
                continue
            track_attrs = score.tracks[tp.id].attributes
            if tp.id in full_ar_ids:
                # Full-AR per-bar overrides flow through
                # BarAttributeValueConstraint in _build_constraints, not the
                # prompt. Skip the prompt-write path here.
                continue
            # Resolve attribute name -> token_type via analyzer (the same
            # key shape used by compute_bar_tokens above).
            for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
                for attr_name, val in (bar_dict or {}).items():
                    attr_obj = (
                        analyzer.get(attr_name) if (analyzer and hasattr(analyzer, "get")) else None
                    )
                    if attr_obj is None:
                        continue
                    track_attrs[f"bar_{attr_obj.token_type}_{int(bar_idx)}"] = int(val)
            # Per-bar controls — map control name to its token-type string.
            for bar_idx, bar_dict in (tp.bar_controls or {}).items():
                for ctrl_name, val in (bar_dict or {}).items():
                    if ctrl_name == "time_signature":
                        track_attrs[f"bar_TimeSig_{int(bar_idx)}"] = int(val)

        # Classify every bar in the step window per the request: a bar is
        # CONTEXT unless it is a generation target (step.bars_to_generate) or
        # listed in tp.mask_bars. The C++ planner's AR branch sets ctx=False
        # for every non-yet-generated bar on AR tracks (so suffix-AR prefix
        # bars would otherwise be encoded as MASK_BAR); this enforces the
        # correct semantic: not-marked = not-masked.
        gen_set = set(step.bars_to_generate)  # {(track_id, bar_abs), …}
        new_ctx = [list(row) for row in step.context]
        ctx_dirty = False
        req_ids = {tp.id: set(tp.mask_bars) for tp in self._request.tracks}
        # Ignored tracks must never be re-classified as context here — an
        # ignored track has bars=[] and no mask_bars, so every one of its
        # bars trivially satisfies "not generating, not masked" and this
        # loop would otherwise force want_ctx=True, overwriting whatever
        # the planner already correctly set from mask.ignore (see
        # _build_selection_mask) and putting the "ignored" track's real
        # content right back into the model's context.
        ignored_ids = {tp.id for tp in self._request.tracks if tp.ignore}
        for t_idx, row in enumerate(new_ctx):
            if t_idx not in req_ids:
                continue
            if t_idx in ignored_ids:
                if any(row[step.start_bar:step.end_bar]):
                    for b_abs in range(step.start_bar, min(step.end_bar, len(row))):
                        row[b_abs] = False
                    ctx_dirty = True
                continue
            masked = req_ids[t_idx]
            for b_abs in range(step.start_bar, step.end_bar):
                if b_abs >= len(row):
                    continue
                want_ctx = (t_idx, b_abs) not in gen_set and b_abs not in masked
                if row[b_abs] != want_ctx:
                    row[b_abs] = want_ctx
                    ctx_dirty = True
        if ctx_dirty:
            step.context = new_ctx

        # Emit the prompt-state snapshot ONCE per run() — first step only.
        # During AR later steps would flip T→C as bars get committed; the
        # caller (e.g. studio) has a single static prediction to compare
        # against, so later snapshots would never line up.
        if emit_snapshot and self.prompt_state_sink is not None and not self._snapshot_sent:
            self._snapshot_sent = True
            try:
                self.prompt_state_sink(self._snapshot_prompt_state(step))
            except Exception as exc:
                log.warning("prompt_state_sink failed: %s", exc)

        # Window size flows via EncodeOptions.window_bars, set inside
        # SessionState from the step. Nothing to plumb through attributes.
        t_enc = time.perf_counter()
        mask_mode = getattr(self._request.config, "mask_mode", "token")
        use_span_masks = mask_mode in ("attention", "attention_approx", "attention_skip")

        # Piece-level controls: injected into EncodeOptions so piece-level tokens
        # (UseVelocity, UseMicrotiming, Genre) carry the correct value in context.
        piece_controls = getattr(self._request, "controls", {}) or {}
        _use_velocity    = (1 if piece_controls["velocity"]    else 0) if "velocity"    in piece_controls else -1
        _use_microtiming = (1 if piece_controls["microtiming"] else 0) if "microtiming" in piece_controls else -1
        _genre = -1
        if "genre" in piece_controls:
            grouping = self._engine._tokenizer._vocab.config().genre_grouping
            if grouping is not None:
                _genre = grouping.encode(piece_controls["genre"])

        score = self._engine._tokenizer.normalize_input(score)
        state = _core.SessionState(
            to_cpp(score), step,
            self._engine._tokenizer._vocab,
            self._build_constraints(step),
            self._engine._tokenizer._encoder,
            self._engine._tokenizer._decoder,
            use_span_masks=use_span_masks,
            remove_future_bars=(mask_mode == "remove"),
            use_velocity=_use_velocity,
            use_microtiming=_use_microtiming,
            genre=_genre,
        )
        return score, state, mask_mode, use_span_masks, t_enc, _use_velocity, _use_microtiming

    def _sample_step(self, score: Score, step, temperature: float, seed: int = -1) -> Score:
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]") from None

        # Seed the global torch RNG so torch.multinomial below is reproducible
        # when the user pins a seed. seed<0 means "leave the RNG state alone"
        # — i.e. continue from wherever the global generator currently is.
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        score, state, mask_mode, use_span_masks, t_enc, _use_velocity, _use_microtiming = self._prepare_state(score, step)

        _ctx_toks = state.context_tokens()
        context_len = len(_ctx_toks)
        self.peak_window = self.peak_window_tokens()   # the step before this one
        self.context_len = context_len
        self.step_gen_start = self.gen_count
        spans = state.hidden_spans() if use_span_masks else []

        # Log context token sequence so we can verify what the model sees
        _tlogger = getattr(self, "token_logger", None)
        if _tlogger is not None:
            _tlogger.log_context(list(_ctx_toks), self._engine._tokenizer._vocab)

        # Detect whether velocity was used in this context encoding by looking
        # for any VelocityLevel token. (There is no UseVelocity gate token in
        # the yellow.pt vocab — velocity is implicit from note encoding style.)
        # Recorded in step_summary so cross-scoring can force the same mode.
        _use_vel_actual = 0
        for _vtid in _ctx_toks:
            try:
                if self._engine._tokenizer._vocab.get_type(_vtid).name == "VelocityLevel":
                    _use_vel_actual = 1
                    break
            except Exception:
                pass

        # ── attention: pre-allocate bool buffer on model device (no per-token cat) ──
        if spans and mask_mode == "attention":
            _kv_dev = next(self._engine._model.parameters()).device
            _kv_cap = self._model_max_context()
            _kv_buf = torch.ones(_kv_cap, dtype=torch.bool, device=_kv_dev)
            for s, e in spans:
                _kv_buf[s:e] = False
            _kv_len = context_len
        else:
            _kv_buf = None
            _kv_len = 0

        # ── attention_approx: one-shot prefill mask (CPU, single use) ──
        if spans and mask_mode == "attention_approx":
            _prefill_mask = torch.ones(context_len, dtype=torch.bool)
            for s, e in spans:
                _prefill_mask[s:e] = False
        else:
            _prefill_mask = None
        _approx_surgery_done = False

        # ── attention_skip: build filtered token sequence and skip position_ids ──
        if spans and mask_mode == "attention_skip":
            _all_ctx = state.context_tokens()
            _keep = [True] * len(_all_ctx)
            for s, e in spans:
                for i in range(s, e):
                    _keep[i] = False
            _skip_ids = [t for t, k in zip(_all_ctx, _keep, strict=False) if k]
            _skip_pos = [i for i, k in enumerate(_keep) if k]
            _next_pos = _skip_pos[-1] + 1 if _skip_pos else context_len
            _skip_ctx = torch.tensor([_skip_ids], dtype=torch.long)
            _skip_pos_t = torch.tensor([_skip_pos], dtype=torch.long)
        else:
            _skip_ctx = _skip_pos_t = None
            _next_pos = 0

        if self.enable_profiling:
            self.encode_time += time.perf_counter() - t_enc
        model_max_ctx = self._model_max_context()
        self.max_context_len = model_max_ctx
        if context_len >= model_max_ctx:
            raise _ContextOverflow(context_len, model_max_ctx)
        max_gen_tokens = model_max_ctx - context_len - 1
        # gen_count runs across every step and attempt of the session; the
        # budget is this attempt's own tokens.
        gen_start = self.gen_count
        n_gen_tracks = sum(1 for tp in self._request.tracks
                           if not getattr(tp, "ignore", False) and tp.bars)
        if max_gen_tokens < n_gen_tracks * 20:
            import warnings
            warnings.warn(
                f"Context nearly full: {context_len}/{model_max_ctx} tokens, "
                f"only {max_gen_tokens} tokens left for {n_gen_tracks} generating tracks "
                f"(~{max_gen_tokens // max(n_gen_tracks, 1)} per track). "
                f"Late tracks may be silently truncated.",
                stacklevel=3,
            )

        # Pre-allocate mask buffer once for this step (reused every token).
        # Deliberately kept on CPU, NOT the model device: on MPS every one of
        # these per-token ops (mask copy, masked_fill, softmax, cumsum, sort,
        # searchsorted, multinomial) pays a GPU dispatch/sync cost that swamps
        # their actual (trivial) compute cost — only the batched forward pass
        # in kv.forward() is big enough to be worth GPU dispatch. `logits` is
        # pulled to CPU right after the forward pass below (one small D2H
        # transfer/token) so the whole grammar-mask + sampling pipeline runs
        # on CPU regardless of model device.
        vocab_size = self._engine._tokenizer.vocab_size()
        mask_buf = torch.empty(vocab_size, dtype=torch.bool, device="cpu")
        # Periodic cache release for very long single attempts — see the
        # per-request empty_cache() in http/server.py for the fuller
        # rationale (MPS allocator fragmentation from the growing KV cache).
        _device_is_mps = next(self._engine._model.parameters()).device.type == "mps"

        # Pre-compute token-ID ranges for hard vocab blocks.
        # When microtiming=False, Delta+DeltaDirection tokens are force-zeroed
        # in mask_buf after the grammar mask loads — they never enter the
        # sampling distribution regardless of model conditioning.
        # Same for VelocityLevel when velocity=False.
        def _tok_range(tt: "_core.TokenType"):
            s, e = self._engine._tokenizer._vocab.range(tt)
            return (s, e) if s >= 0 and e > s else None

        _block_delta = _use_microtiming == 0
        _block_vel   = _use_velocity == 0
        _delta_range     = _tok_range(_core.TokenType.Delta)          if _block_delta else None
        _delta_dir_range = _tok_range(_core.TokenType.DeltaDirection) if _block_delta else None
        _vel_range       = _tok_range(_core.TokenType.VelocityLevel)  if _block_vel   else None

        # Pitch scale/register control (controls["pitch_mask"]) — hard
        # allow-set ANDed into mask_buf below, soft shape reweighted into
        # probs right before the sampling filters.
        _pitch_range, _pitch_allow, _pitch_shape = self._build_pitch_mask(step)

        # Rhythm control (controls["rhythm_mask"]) — hard exact-position mode
        # drives a _RhythmForcer that narrows mask_buf every token; soft grid
        # mode reweights probs like _pitch_shape.
        _rhythm_range, _rhythm_schedule, _rhythm_grid = self._build_rhythm_mask(step)
        _rhythm_forcer = None
        if _rhythm_schedule is not None:
            _barend_range = _tok_range(_core.TokenType.BarEnd)
            _onset_range = _tok_range(_core.TokenType.NoteOnset)
            _vel_range_any = _tok_range(_core.TokenType.VelocityLevel)
            _rhythm_forcer = _RhythmForcer(
                _rhythm_schedule, _rhythm_range, _onset_range, _vel_range_any, _barend_range[0]
            )

        # Variation/remix (controls["remix"]) — forces the reference score's
        # onset schedule exactly (like a rhythm_mask derived from it) while
        # independently resampling p% of note-level values. Mutually
        # exclusive with rhythm_mask on the same track (validation.py), so
        # at most one of _rhythm_forcer / _variation_forcer is ever active.
        _remix_mode, _remix_amount, _remix_bars = self._build_remix_schedule(score, step)
        _variation_forcer = None
        if _remix_bars is not None:
            _pos_range_r = _tok_range(_core.TokenType.TimeAbsolutePos)
            _barend_range_r = _tok_range(_core.TokenType.BarEnd)
            _onset_range_r = _tok_range(_core.TokenType.NoteOnset)
            _dur_range_r = _tok_range(_core.TokenType.NoteDuration)
            _vel_range_r = _tok_range(_core.TokenType.VelocityLevel)
            _variation_forcer = _VariationForcer(
                _remix_mode, _remix_amount, _remix_bars,
                _pos_range_r, _onset_range_r, _dur_range_r, _vel_range_r, _barend_range_r[0],
                count_on_onset=self._step_is_drums(score, step),
            )

        # Baseline for the note_sink diff: decode BEFORE any tokens are
        # generated (context + not-yet-generated target bars, however the
        # decoder resolves those — typically empty). Without this, the
        # first diff would have no baseline to compare against and either
        # lose the first note entirely or flood the stream with every
        # pre-existing context note mislabeled as "new".
        _last_streamed_partial = (
            self._restore_window(score, self._engine._tokenizer.normalize_output(from_cpp(state.result())), step)
            if self.note_sink is not None else None
        )
        _last_stream_t = time.perf_counter()
        kv = _KVRunner(self._engine._model, self._engine._initial_kv)
        with torch.no_grad():
            while (
                not state.complete()
                and self.gen_count - gen_start < max_gen_tokens
                and not (self.cancel_event is not None and self.cancel_event.is_set())
            ):
                is_prefill = kv.is_prefill

                # Build ctx and determine key_mask / position_ids for this step
                if is_prefill:
                    if mask_mode == "attention_skip" and _skip_ctx is not None:
                        ctx = _skip_ctx
                        km = None
                        pos_ids = _skip_pos_t
                    else:
                        ctx = torch.tensor([state.context_tokens()], dtype=torch.long)
                        km = (
                            _kv_buf[:_kv_len]
                            if mask_mode == "attention" and _kv_buf is not None
                            else _prefill_mask
                            if mask_mode == "attention_approx" and _prefill_mask is not None
                            else None
                        )
                        pos_ids = None
                else:
                    ctx = torch.tensor([[state.context_tokens()[-1]]], dtype=torch.long)
                    km = (
                        _kv_buf[:_kv_len]
                        if mask_mode == "attention" and _kv_buf is not None
                        else None
                    )
                    pos_ids = (
                        torch.tensor([[_next_pos]], dtype=torch.long)
                        if mask_mode == "attention_skip" and _skip_ctx is not None
                        else None
                    )

                t_fwd = time.perf_counter()
                logits_seq = kv.forward(ctx, key_mask=km, position_ids=pos_ids)
                if logits_seq.device.type == "mps":
                    torch.mps.synchronize()
                if self.enable_profiling:
                    self.model_forward_time += time.perf_counter() - t_fwd

                # attention_approx: KV surgery after prefill — null masked positions
                if (
                    mask_mode == "attention_approx"
                    and is_prefill
                    and not _approx_surgery_done
                    and spans
                ):
                    kv.null_positions(spans)
                    _approx_surgery_done = True

                # Single small D2H transfer/token — everything downstream
                # (grammar mask, softmax, sampling filters, multinomial) runs
                # on CPU from here on. See mask_buf comment above for why.
                logits = logits_seq[0, -1].to("cpu", non_blocking=False)

                # Reuse pre-allocated bool buffer for grammar mask
                mask_buf.copy_(torch.as_tensor(state.logit_mask(), dtype=torch.bool))

                # Hard vocab blocks: force token types to -inf regardless of
                # grammar state (microtiming=False → no Delta/DeltaDirection,
                # velocity=False → no VelocityLevel).
                if _delta_range is not None:
                    mask_buf[_delta_range[0]:_delta_range[1]] = False
                if _delta_dir_range is not None:
                    mask_buf[_delta_dir_range[0]:_delta_dir_range[1]] = False
                if _vel_range is not None:
                    mask_buf[_vel_range[0]:_vel_range[1]] = False
                if _pitch_allow is not None:
                    lo, hi = _pitch_range
                    mask_buf[lo:hi] &= _pitch_allow
                if _rhythm_forcer is not None:
                    _rhythm_forcer.apply(mask_buf)
                if _variation_forcer is not None:
                    _variation_forcer.apply(mask_buf)

                n_legal = int(mask_buf.sum().item())
                if n_legal == 0:
                    if _tlogger is not None:
                        _tlogger.log_grammar_collapse(self.gen_count, 0)
                    raise RuntimeError(
                        "sampling crashed: constraint graph error "
                        "(zero legal tokens at this step — over-constrained "
                        "attribute values or incompatible grammar state)"
                    )
                masked_logits = logits.masked_fill(~mask_buf, float("-inf"))
                # T=1 distribution after grammar mask — the model's true belief.
                # Log this before temperature/sampling-filter shaping.
                _raw_probs = masked_logits.softmax(-1)

                probs = (masked_logits / temperature).softmax(-1)

                if torch.isnan(probs.sum()) or probs.sum() < 1e-6:
                    # The model's distribution collapsed onto masked-out tokens.
                    # Do NOT disable the grammar — pick uniformly over the legal
                    # set so the sequence stays valid (decoder won't drop notes).
                    logging.debug(
                        f"  grammar-collapse: model probs vanish under "
                        f"mask (n_legal={n_legal}); sampling uniformly "
                        f"from legal tokens"
                    )
                    if _tlogger is not None:
                        _tlogger.log_grammar_collapse(self.gen_count, n_legal)
                    probs = mask_buf.to(torch.float32)
                    probs = probs / probs.sum()

                if _pitch_shape is not None:
                    lo, hi = _pitch_range
                    probs = probs.clone()
                    probs[lo:hi] = probs[lo:hi] * _pitch_shape
                    _shape_total = float(probs.sum().item())
                    if _shape_total > 0:
                        probs = probs / _shape_total

                if _rhythm_grid is not None:
                    lo, hi = _rhythm_range
                    probs = probs.clone()
                    probs[lo:hi] = probs[lo:hi] * _rhythm_grid
                    _shape_total = float(probs.sum().item())
                    if _shape_total > 0:
                        probs = probs / _shape_total

                probs = self._apply_sampling_filters(probs)

                token = torch.multinomial(probs, 1).item()
                # Diagnostic: when we just sampled a melodic NoteOnset, check
                # whether the very next mask permits non-NoteDuration tokens
                # (a real-grammar bug if so).
                try:
                    tname = self._engine._tokenizer._vocab.get_type(token).name
                    _, tval = self._engine._tokenizer._vocab.decode(token)
                except Exception:
                    tname, tval = "?", -1
                if _rhythm_forcer is not None:
                    _rhythm_forcer.advance(tname)
                if _variation_forcer is not None:
                    _variation_forcer.advance(tname)

                # Log this token (T=1 raw probs, grammar mask, model top-5)
                if _tlogger is not None:
                    _tlogger.log_token(
                        self.gen_count, token, tname, tval,
                        _raw_probs.cpu(), mask_buf.cpu(),
                        self._engine._tokenizer._vocab,
                    )

                if tname == "NoteOnset" and logging.getLogger().isEnabledFor(logging.DEBUG):
                    state.advance(token)
                    nxt_mask = list(state.logit_mask())
                    legal_types = set()
                    for i, ok in enumerate(nxt_mask):
                        if ok:
                            try:
                                legal_types.add(self._engine._tokenizer._vocab.get_type(i).name)
                            except Exception:
                                pass
                    logging.debug(
                        f"  after NoteOnset({token}) n_legal={sum(nxt_mask)} "
                        f"legal_types={sorted(legal_types)}"
                    )
                    self.gen_count += 1
                    if mask_mode == "attention" and _kv_buf is not None:
                        _kv_len += 1
                    elif mask_mode == "attention_skip" and _skip_ctx is not None:
                        _next_pos += 1
                    continue
                state.advance(token)
                self.gen_count += 1
                if _device_is_mps and self.gen_count % 200 == 0:
                    torch.mps.empty_cache()
                if self.progress_sink is not None and self.gen_count % PROGRESS_EVERY == 0:
                    self.progress_sink()
                # attention: extend buffer pointer for generated token
                if mask_mode == "attention" and _kv_buf is not None:
                    _kv_len += 1
                # attention_skip: advance next position_id
                elif mask_mode == "attention_skip" and _skip_ctx is not None:
                    _next_pos += 1

                # Streaming: a note is fully formed the instant its
                # NoteDuration token lands (pitch + duration both known,
                # velocity already applied for anything preceding it in the
                # decode). Re-decode the partial score and diff against the
                # last streamed snapshot rather than hand-tracking
                # pitch/position/velocity ourselves — reuses the same
                # decode path a natural completion uses, so it can't drift
                # out of sync with what the final result will actually say.
                # NoteOnset too: a drum note has no NoteDuration, so drum-only steps streamed nothing.
                if (self.note_sink is not None and tname in ("NoteDuration", "NoteOnset")
                        and time.perf_counter() - _last_stream_t >= STREAM_EVERY_S):
                    partial = self._restore_window(
                        score, self._engine._tokenizer.normalize_output(from_cpp(state.result())), step
                    )
                    new_notes = self._diff_notes(_last_streamed_partial, partial)
                    if new_notes:
                        self.note_sink(new_notes)
                    _last_streamed_partial = partial
                    _last_stream_t = time.perf_counter()

        # Loop exited either because state.complete() (natural stop — the
        # grammar reached a valid end, e.g. all requested bars filled) or
        # because gen_count hit max_gen_tokens (context budget ran out
        # first). Distinguish them: a truncated generation may have stopped
        # mid-bar/mid-track, not because it was actually "done".
        self.truncated = not state.complete()
        if self.truncated:
            log.warning(
                "generation truncated: hit max_gen_tokens=%d before the grammar "
                "reached a natural stop (context_len=%d)",
                max_gen_tokens, context_len,
            )

        # Step summary: token-type counts, context size, grammar collapses
        if _tlogger is not None:
            _tlogger.log_step_summary(context_len, max_gen_tokens, n_gen_tracks,
                                      request=self._request,
                                      use_velocity=_use_vel_actual)

        t_dec = time.perf_counter()
        result = self._restore_window(
            score, self._engine._tokenizer.normalize_output(from_cpp(state.result())), step
        )
        if self.enable_profiling:
            self.decode_time += time.perf_counter() - t_dec
        if self.note_sink is not None:   # whatever was generated since the last decode
            new_notes = self._diff_notes(_last_streamed_partial, result)
            if new_notes:
                self.note_sink(new_notes)

        # Bar summaries from the decoded result (all tracks, all bars)
        if _tlogger is not None:
            gap_bar_set = set()
            for tp in self._request.tracks:
                if not getattr(tp, "ignore", False) and tp.bars:
                    gap_bar_set.update(tp.bars)
            _tlogger.log_bar_summaries(result, self._request, sorted(gap_bar_set),
                                       tpb_spq=0)  # tpb_spq unused (bar index used directly)

        if self.cancel_event is not None and self.cancel_event.is_set():
            log.info("generation cancelled (gen_count=%d)", self.gen_count)
            raise GenerationCancelled(result, self.gen_count)

        return result

    def _sample_step_batch(
        self, score: Score, step, temperatures: list, seeds: list
    ) -> list:
        """Batched variant of _sample_step: runs len(temperatures) independent
        generation attempts (one row per attempt, same prompt, own
        temperature/seed/grammar-state each) as ONE forward pass per token
        step instead of len(temperatures) sequential single-row passes.

        Turns a latency-bound sequential decode (can't fill a GPU at
        batch=1) into a throughput-bound batched decode. Scoped to requests
        with zero hidden spans — raises _BatchNotEligible otherwise, and the
        caller (_run_step) falls back to the sequential path.

        Returns a list of
        (candidate_score_or_None, gen_count, error_or_None, truncated)
        tuples, one per row, in the same order as `temperatures`/`seeds`.
        truncated=True means this row hit max_gen_tokens (context budget)
        before its grammar reached a natural stop — its content may end
        mid-bar/mid-track rather than because it was actually finished.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]") from None

        B = len(temperatures)

        # ---- Setup below mirrors _sample_step exactly, run ONCE: it's a
        # pure function of (score, request), independent of seed/temperature,
        # so redoing it per-row (like the sequential path effectively does,
        # once per attempt) would be pointless duplicated work. ----
        full_ar_ids: set[int] = set()
        for tp in self._request.tracks:
            if tp.autoregressive and (not tp.bars or min(tp.bars) == 0):
                full_ar_ids.add(tp.id)
        self._full_ar_ids = full_ar_ids
        analyzer = self._engine._analyzer
        if analyzer is not None:
            for t_idx, track in enumerate(score.tracks):
                if t_idx in full_ar_ids:
                    continue
                new_attrs = dict(track.attributes)
                new_attrs.update(analyzer.compute_track_tokens(score, t_idx))
                for b_idx in range(len(track.bars)):
                    for k, v in analyzer.compute_bar_tokens(score, t_idx, b_idx).items():
                        new_attrs[f"bar_{k}_{b_idx}"] = v
                track.attributes = new_attrs

        for tp in self._request.tracks:
            if tp.id < len(score.tracks) and tp.id not in full_ar_ids:
                score.tracks[tp.id].attributes.update(tp.attributes)

        for tp in self._request.tracks:
            if tp.id >= len(score.tracks):
                continue
            track_attrs = score.tracks[tp.id].attributes
            if tp.id in full_ar_ids:
                continue
            for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
                for attr_name, val in (bar_dict or {}).items():
                    attr_obj = (
                        analyzer.get(attr_name) if (analyzer and hasattr(analyzer, "get")) else None
                    )
                    if attr_obj is None:
                        continue
                    track_attrs[f"bar_{attr_obj.token_type}_{int(bar_idx)}"] = int(val)
            for bar_idx, bar_dict in (tp.bar_controls or {}).items():
                for ctrl_name, val in (bar_dict or {}).items():
                    if ctrl_name == "time_signature":
                        track_attrs[f"bar_TimeSig_{int(bar_idx)}"] = int(val)

        gen_set = set(step.bars_to_generate)
        new_ctx = [list(row) for row in step.context]
        ctx_dirty = False
        req_ids = {tp.id: set(tp.mask_bars) for tp in self._request.tracks}
        # See the matching comment in _sample_step — ignored tracks must
        # stay excluded from context here, not get re-classified as True.
        ignored_ids = {tp.id for tp in self._request.tracks if tp.ignore}
        for t_idx, row in enumerate(new_ctx):
            if t_idx not in req_ids:
                continue
            if t_idx in ignored_ids:
                if any(row[step.start_bar:step.end_bar]):
                    for b_abs in range(step.start_bar, min(step.end_bar, len(row))):
                        row[b_abs] = False
                    ctx_dirty = True
                continue
            masked = req_ids[t_idx]
            for b_abs in range(step.start_bar, step.end_bar):
                if b_abs >= len(row):
                    continue
                want_ctx = (t_idx, b_abs) not in gen_set and b_abs not in masked
                if row[b_abs] != want_ctx:
                    row[b_abs] = want_ctx
                    ctx_dirty = True
        if ctx_dirty:
            step.context = new_ctx

        if self.prompt_state_sink is not None and not self._snapshot_sent:
            self._snapshot_sent = True
            try:
                self.prompt_state_sink(self._snapshot_prompt_state(step))
            except Exception as exc:
                log.warning("prompt_state_sink failed: %s", exc)

        t_enc = time.perf_counter()
        mask_mode = getattr(self._request.config, "mask_mode", "token")
        use_span_masks = mask_mode in ("attention", "attention_approx", "attention_skip")

        piece_controls = getattr(self._request, "controls", {}) or {}
        _use_velocity    = (1 if piece_controls["velocity"]    else 0) if "velocity"    in piece_controls else -1
        _use_microtiming = (1 if piece_controls["microtiming"] else 0) if "microtiming" in piece_controls else -1
        _genre = -1
        if "genre" in piece_controls:
            grouping = self._engine._tokenizer._vocab.config().genre_grouping
            if grouping is not None:
                _genre = grouping.encode(piece_controls["genre"])

        score = self._engine._tokenizer.normalize_input(score)

        # One independent grammar-state machine per row — .advance() mutates
        # it in place, so rows can't share an instance once sampling diverges
        # (which happens immediately: different seeds -> different tokens).
        # Constraint graphs are stateful too (track constraint-satisfaction
        # progress as tokens are advanced), so each row needs its OWN
        # instance as well — a shared one would get corrupted by interleaved
        # .advance() calls from different rows, exactly like the shared
        # SessionState issue above.
        states = [
            _core.SessionState(
                to_cpp(copy.deepcopy(score)), step,
                self._engine._tokenizer._vocab,
                self._build_constraints(step),
                self._engine._tokenizer._encoder,
                self._engine._tokenizer._decoder,
                use_span_masks=use_span_masks,
                remove_future_bars=(mask_mode == "remove"),
                use_velocity=_use_velocity,
                use_microtiming=_use_microtiming,
                genre=_genre,
            )
            for _ in range(B)
        ]

        # Batching only covers the "nothing hidden" case (see class docstring
        # on _BatchNotEligible) — bail out to the sequential path otherwise.
        spans = states[0].hidden_spans() if use_span_masks else []
        if spans:
            raise _BatchNotEligible(
                f"{len(spans)} hidden span(s) for mask_mode={mask_mode!r}; "
                "batched span-masking isn't implemented"
            )

        _ctx_toks = states[0].context_tokens()  # identical across rows: same prompt
        context_len = len(_ctx_toks)
        self.peak_window = self.peak_window_tokens()   # the step before this one
        self.context_len = context_len
        self.step_gen_start = self.gen_count

        _tlogger = getattr(self, "token_logger", None)
        if _tlogger is not None:
            _tlogger.log_context(list(_ctx_toks), self._engine._tokenizer._vocab)

        _use_vel_actual = 0
        for _vtid in _ctx_toks:
            try:
                if self._engine._tokenizer._vocab.get_type(_vtid).name == "VelocityLevel":
                    _use_vel_actual = 1
                    break
            except Exception:
                pass

        if self.enable_profiling:
            self.encode_time += time.perf_counter() - t_enc
        model_max_ctx = self._model_max_context()
        self.max_context_len = model_max_ctx
        if context_len >= model_max_ctx:
            raise _ContextOverflow(context_len, model_max_ctx)
        max_gen_tokens = model_max_ctx - context_len - 1
        n_gen_tracks = sum(1 for tp in self._request.tracks
                           if not getattr(tp, "ignore", False) and tp.bars)
        if max_gen_tokens < n_gen_tracks * 20:
            import warnings
            warnings.warn(
                f"Context nearly full: {context_len}/{model_max_ctx} tokens, "
                f"only {max_gen_tokens} tokens left for {n_gen_tracks} generating tracks "
                f"(~{max_gen_tokens // max(n_gen_tracks, 1)} per track). "
                f"Late tracks may be silently truncated.",
                stacklevel=3,
            )

        vocab_size = self._engine._tokenizer.vocab_size()
        mask_buf = torch.empty(vocab_size, dtype=torch.bool, device="cpu")
        _device_is_mps = next(self._engine._model.parameters()).device.type == "mps"

        def _tok_range(tt: "_core.TokenType"):
            s, e = self._engine._tokenizer._vocab.range(tt)
            return (s, e) if s >= 0 and e > s else None

        _block_delta = _use_microtiming == 0
        _block_vel   = _use_velocity == 0
        _delta_range     = _tok_range(_core.TokenType.Delta)          if _block_delta else None
        _delta_dir_range = _tok_range(_core.TokenType.DeltaDirection) if _block_delta else None
        _vel_range       = _tok_range(_core.TokenType.VelocityLevel)  if _block_vel   else None

        # Pitch scale/register control — see the matching comment in
        # _sample_step. Computed once for the whole batch: every row shares
        # the same prompt/step, only seed/temperature differ per row.
        _pitch_range, _pitch_allow, _pitch_shape = self._build_pitch_mask(step)

        # Rhythm control — see the matching comment in _sample_step. One
        # _RhythmForcer PER ROW: rows are otherwise expected to stay in
        # lockstep on a deterministic schedule, but a row that dies early
        # (row_errors[i] set) must not corrupt another row's progress.
        _rhythm_range, _rhythm_schedule, _rhythm_grid = self._build_rhythm_mask(step)
        _rhythm_forcers = [None] * B
        if _rhythm_schedule is not None:
            _barend_range = _tok_range(_core.TokenType.BarEnd)
            _onset_range = _tok_range(_core.TokenType.NoteOnset)
            _vel_range_any = _tok_range(_core.TokenType.VelocityLevel)
            _rhythm_forcers = [
                _RhythmForcer(_rhythm_schedule, _rhythm_range, _onset_range, _vel_range_any, _barend_range[0])
                for _ in range(B)
            ]

        # Variation/remix — see the matching comment in _sample_step. One
        # _VariationForcer PER ROW (each row rolls its own force/resample
        # decisions independently — that's the entire point of a batch of
        # candidates for the same remix request).
        _remix_mode, _remix_amount, _remix_bars = self._build_remix_schedule(score, step)
        _variation_forcers = [None] * B
        if _remix_bars is not None:
            _pos_range_r = _tok_range(_core.TokenType.TimeAbsolutePos)
            _barend_range_r = _tok_range(_core.TokenType.BarEnd)
            _onset_range_r = _tok_range(_core.TokenType.NoteOnset)
            _dur_range_r = _tok_range(_core.TokenType.NoteDuration)
            _vel_range_r = _tok_range(_core.TokenType.VelocityLevel)
            _variation_forcers = [
                _VariationForcer(
                    _remix_mode, _remix_amount, _remix_bars,
                    _pos_range_r, _onset_range_r, _dur_range_r, _vel_range_r, _barend_range_r[0],
                    count_on_onset=self._step_is_drums(score, step),
                )
                for _ in range(B)
            ]

        # Independent RNG stream per row, seeded once — NOT re-seeded every
        # token. torch.multinomial(..., generator=g) draws from g's own
        # advancing stream, so rows don't interfere with each other despite
        # sharing one Python-level token loop. seed<0 -> None -> falls back
        # to the shared global generator (matches the sequential path's
        # "seed<0 leaves RNG state alone, don't-care-about-reproducibility"
        # semantics).
        row_generators = [
            torch.Generator().manual_seed(int(seeds[i])) if seeds[i] is not None and seeds[i] >= 0 else None
            for i in range(B)
        ]

        kv = _KVRunner(self._engine._model, self._engine._initial_kv)
        done = [False] * B
        gen_counts = [0] * B
        row_errors = [None] * B
        last_tokens = [None] * B

        with torch.no_grad():
            is_prefill = True
            while (
                not all(done)
                and max(gen_counts) < max_gen_tokens
                and not (self.cancel_event is not None and self.cancel_event.is_set())
            ):
                if is_prefill:
                    ctx = torch.tensor([_ctx_toks], dtype=torch.long).repeat(B, 1)
                else:
                    # Finished rows keep getting fed their own last token —
                    # wasted compute for them, but keeps the batch shape
                    # fixed and simple; their output is discarded below.
                    ctx = torch.tensor(
                        [[last_tokens[i] if last_tokens[i] is not None else 0] for i in range(B)],
                        dtype=torch.long,
                    )

                t_fwd = time.perf_counter()
                logits_seq = kv.forward(ctx)
                if logits_seq.device.type == "mps":
                    torch.mps.synchronize()
                if self.enable_profiling:
                    self.model_forward_time += time.perf_counter() - t_fwd
                is_prefill = False

                # One D2H transfer for the whole batch's last-position logits.
                logits_batch = logits_seq[:, -1, :].to("cpu", non_blocking=False)

                for i in range(B):
                    if done[i]:
                        continue
                    mask_buf.copy_(torch.as_tensor(states[i].logit_mask(), dtype=torch.bool))
                    if _delta_range is not None:
                        mask_buf[_delta_range[0]:_delta_range[1]] = False
                    if _delta_dir_range is not None:
                        mask_buf[_delta_dir_range[0]:_delta_dir_range[1]] = False
                    if _vel_range is not None:
                        mask_buf[_vel_range[0]:_vel_range[1]] = False
                    if _pitch_allow is not None:
                        lo, hi = _pitch_range
                        mask_buf[lo:hi] &= _pitch_allow
                    if _rhythm_forcers[i] is not None:
                        _rhythm_forcers[i].apply(mask_buf)
                    if _variation_forcers[i] is not None:
                        _variation_forcers[i].apply(mask_buf)

                    n_legal = int(mask_buf.sum().item())
                    if n_legal == 0:
                        if _tlogger is not None:
                            _tlogger.log_grammar_collapse(gen_counts[i], 0)
                        row_errors[i] = (
                            "sampling crashed: constraint graph error "
                            "(zero legal tokens at this step — over-constrained "
                            "attribute values or incompatible grammar state)"
                        )
                        done[i] = True
                        continue

                    masked_logits = logits_batch[i].masked_fill(~mask_buf, float("-inf"))
                    probs = (masked_logits / temperatures[i]).softmax(-1)

                    if torch.isnan(probs.sum()) or probs.sum() < 1e-6:
                        if _tlogger is not None:
                            _tlogger.log_grammar_collapse(gen_counts[i], n_legal)
                        probs = mask_buf.to(torch.float32)
                        probs = probs / probs.sum()

                    if _pitch_shape is not None:
                        lo, hi = _pitch_range
                        probs = probs.clone()
                        probs[lo:hi] = probs[lo:hi] * _pitch_shape
                        _shape_total = float(probs.sum().item())
                        if _shape_total > 0:
                            probs = probs / _shape_total

                    if _rhythm_grid is not None:
                        lo, hi = _rhythm_range
                        probs = probs.clone()
                        probs[lo:hi] = probs[lo:hi] * _rhythm_grid
                        _shape_total = float(probs.sum().item())
                        if _shape_total > 0:
                            probs = probs / _shape_total

                    probs = self._apply_sampling_filters(probs)

                    gen = row_generators[i]
                    token = (
                        torch.multinomial(probs, 1, generator=gen).item()
                        if gen is not None
                        else torch.multinomial(probs, 1).item()
                    )

                    if _rhythm_forcers[i] is not None or _variation_forcers[i] is not None or _tlogger is not None:
                        try:
                            tname = self._engine._tokenizer._vocab.get_type(token).name
                            _, tval = self._engine._tokenizer._vocab.decode(token)
                        except Exception:
                            tname, tval = "?", -1
                        if _tlogger is not None:
                            _tlogger.log_token(
                                gen_counts[i], token, tname, tval,
                                probs.cpu(), mask_buf.cpu(),
                                self._engine._tokenizer._vocab,
                            )
                        if _rhythm_forcers[i] is not None:
                            _rhythm_forcers[i].advance(tname)
                        if _variation_forcers[i] is not None:
                            _variation_forcers[i].advance(tname)

                    states[i].advance(token)
                    last_tokens[i] = token
                    gen_counts[i] += 1
                    if states[i].complete():
                        done[i] = True

                self.gen_count = sum(gen_counts)
                if _device_is_mps and self.gen_count % 200 == 0:
                    torch.mps.empty_cache()

        # Per-row: did THIS row's grammar reach a natural stop, or did the
        # batched loop end (globally, once the furthest-along row hit
        # max_gen_tokens) before it got there? A row can be "not done" here
        # for either reason — treat both as truncated (its output may have
        # stopped mid-bar/mid-track, not because it was actually finished).
        row_truncated = [not states[i].complete() for i in range(B)]
        self.truncated = any(row_truncated[i] for i in range(B) if row_errors[i] is None)
        if self.truncated:
            log.warning(
                "run_variations/batch_attempts: %d/%d row(s) truncated (hit "
                "max_gen_tokens=%d before a natural stop)",
                sum(1 for i in range(B) if row_truncated[i] and row_errors[i] is None),
                B, max_gen_tokens,
            )

        if _tlogger is not None:
            _tlogger.log_step_summary(context_len, max_gen_tokens, n_gen_tracks,
                                      request=self._request,
                                      use_velocity=_use_vel_actual)

        t_dec = time.perf_counter()
        results = []
        for i in range(B):
            if row_errors[i] is not None:
                results.append((None, gen_counts[i], row_errors[i], row_truncated[i]))
                continue
            result = self._restore_window(
                score, self._engine._tokenizer.normalize_output(from_cpp(states[i].result())), step
            )
            if _tlogger is not None:
                gap_bar_set = set()
                for tp in self._request.tracks:
                    if not getattr(tp, "ignore", False) and tp.bars:
                        gap_bar_set.update(tp.bars)
                _tlogger.log_bar_summaries(result, self._request, sorted(gap_bar_set), tpb_spq=0)
            results.append((result, gen_counts[i], None, row_truncated[i]))
        if self.enable_profiling:
            self.decode_time += time.perf_counter() - t_dec

        if self.cancel_event is not None and self.cancel_event.is_set():
            log.info("batched generation cancelled (gen_count=%d)", self.gen_count)
            raise GenerationCancelled(results, self.gen_count)

        return results

    def score_from_tokens(
        self,
        token_ids: list,
        step_idx: int = 0,
        use_velocity: int = -1,
    ) -> dict:
        """Teacher-forced log P(token_ids | context from this session).

        Uses a single forward pass over [context_tokens + token_ids], then
        steps the grammar state token-by-token to get per-position masks.
        Returns log P of each token_id under the masked distribution (T=1).

        step_idx: which planner step to use (0 = first/cold-start, which is
                  the most informative for cross-conditional scoring).
        """
        import copy
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]") from None

        cfg  = self._request.config
        mask = self._build_selection_mask()
        import json as _json
        enc_cfg = self._engine._tokenizer._vocab.config()
        try:
            dims = sorted(
                set(_json.loads(enc_cfg.to_json()).get("num_bars_map") or []),
                reverse=True,
            )
        except Exception:
            dims = []
        candidates = [d for d in dims if d <= cfg.model_dim] or [cfg.model_dim]
        enc_cfg.model_dim = candidates[0]

        steps = self._plan(mask, replace_cfg(cfg, model_dim=candidates[0]))
        if step_idx >= len(steps) or not token_ids:
            return {"log_probs": [], "total_logp": 0.0, "n_tokens": 0, "per_type": {}}
        step = steps[step_idx]

        # Build score + SessionState (minimal setup — no analyzer attributes)
        py_score = self._score if isinstance(self._score, Score) else from_cpp(self._score)
        score = self._engine._tokenizer.normalize_input(copy.deepcopy(py_score))

        # _build_constraints uses self._full_ar_ids; infill scoring has none
        self._full_ar_ids = set()
        state = _core.SessionState(
            to_cpp(score), step,
            self._engine._tokenizer._vocab,
            self._build_constraints(step),
            self._engine._tokenizer._encoder,
            self._engine._tokenizer._decoder,
            use_span_masks=False,
            remove_future_bars=False,
            use_velocity=use_velocity, use_microtiming=-1, genre=-1,
        )
        context_tokens = list(state.context_tokens())
        ctx_len = len(context_tokens)

        # Prefill over the full sequence [ctx + token_ids]. Generation never
        # exceeds the positional window (it stops at the budget), but replay
        # scoring can: fall back to sliding-window scoring with stride =
        # half window, so every position keeps at least half a window of
        # left context.
        full_seq = context_tokens + list(token_ids)
        dev = next(self._engine._model.parameters()).device

        def _forward(seq):
            inp = torch.tensor([seq], dtype=torch.long, device=dev)
            with torch.no_grad():
                try:
                    out = self._engine._model(inp)
                except Exception:
                    try:
                        out = self._engine._model(inp, None)
                    except Exception:
                        out = self._engine._model(inp.cpu())
            if not isinstance(out, tuple):
                out = (out,)
            return out[0].cpu()[0]  # [seq_len, vocab]

        n_pos = self._model_max_context()
        if len(full_seq) <= n_pos:
            _logits_seq = _forward(full_seq)

            def _logits_at(p):
                return _logits_seq[p]
        else:
            _stride = max(1, n_pos // 2)
            _window_cache: dict[int, torch.Tensor] = {}

            def _logits_at(p):
                if p < n_pos:
                    start = 0
                else:
                    # smallest stride-aligned start that keeps >= n_pos - stride
                    # tokens of left context for position p
                    start = ((p - n_pos + 1 + _stride - 1) // _stride) * _stride
                if start not in _window_cache:
                    _window_cache[start] = _forward(full_seq[start:start + n_pos])
                return _window_cache[start][p - start]

        vocab = self._engine._tokenizer._vocab
        log_probs: list[float] = []
        per_type: dict[str, list] = {}

        for i, tid in enumerate(token_ids):
            logit_pos = ctx_len - 1 + i
            logits = _logits_at(logit_pos)

            mask_cpu = torch.as_tensor(state.logit_mask(), dtype=torch.bool)
            n_legal = int(mask_cpu.sum().item())
            if n_legal > 0:
                masked_logits = logits.masked_fill(~mask_cpu, float("-inf"))
            else:
                masked_logits = logits

            log_p_dist = torch.log_softmax(masked_logits, dim=-1)
            lp = float(log_p_dist[int(tid)].item())
            log_probs.append(lp)

            try:
                tt = vocab.get_type(int(tid)).name
            except Exception:
                tt = "?"
            per_type.setdefault(tt, []).append(lp)

            try:
                state.advance(int(tid))
            except Exception:
                break

        return {
            "log_probs": log_probs,
            "total_logp": float(sum(log_probs)),
            "n_tokens": len(log_probs),
            "per_type": {
                k: {"mean": sum(v) / len(v), "sum": sum(v), "n": len(v)}
                for k, v in per_type.items()
            },
        }

    def _snapshot_prompt_state(self, step) -> dict:
        """Per-bar prompt state for the current step, post-mask_bars patch.

        Each entry is "C" (CONTEXT), "M" (MASKED via MaskBar token),
        "A" (MASKED via attention span-mask), or "T" (TO_GENERATE).
        "A" only appears when mask_mode is one of "attention*" — purely a
        visualization label; the bar classification is identical to "M".
        """
        ar_ids = {tp.id for tp in self._request.tracks if tp.autoregressive}
        btg = set(step.bars_to_generate)
        mask_mode = getattr(self._request.config, "mask_mode", "token")
        masked_label = (
            "A" if mask_mode in ("attention", "attention_approx", "attention_skip") else "M"
        )
        tracks = []
        for t_idx, row in enumerate(step.context):
            states = []
            for b_abs in range(step.start_bar, step.end_bar):
                if (t_idx, b_abs) in btg:
                    states.append("T")
                elif b_abs < len(row) and row[b_abs]:
                    states.append("C")
                else:
                    states.append(masked_label)
            tracks.append(
                {
                    "id": t_idx,
                    "is_agent": t_idx in ar_ids,
                    "states": states,
                }
            )
        return {
            "start_bar": int(step.start_bar),
            "end_bar": int(step.end_bar),
            "tracks": tracks,
        }

    def _apply_sampling_filters(self, probs):
        """Apply top_k → top_p → mask_k → mask_p in that fixed order.

        Pipeline reasoning: k-filters are rank-based; p-filters are mass-based.
        Applying top filters first narrows the pool to the model's preferred
        region; mask filters then carve out the most-obvious tokens *within*
        that pool — useful for novelty (force the model off its top picks
        while staying inside its high-confidence region).

        All filters mutate a local `keep` boolean mask and return a
        renormalized probability vector. If the pool would become empty, the
        offending filter is skipped (validation catches impossible combos at
        config time; this is a runtime safety net for edge distributions).
        """
        import torch

        cfg = self._request.config
        top_p = float(getattr(cfg, "top_p", 1.0) or 1.0)
        top_k = int(getattr(cfg, "top_k", 0) or 0)
        mask_p = float(getattr(cfg, "mask_p", 0.0) or 0.0)
        mask_k = int(getattr(cfg, "mask_k", 0) or 0)
        if top_p >= 1.0 and top_k <= 0 and mask_p <= 0.0 and mask_k <= 0:
            return probs

        keep = probs > 0
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # number of currently-positive (i.e. legal & non-filtered) tokens, used
        # to size rank-based filters relative to the remaining pool
        legal_n = int(keep.sum().item())

        # top_k: keep only the top_k highest-prob tokens
        if 0 < top_k < legal_n:
            cutoff = sorted_idx[:top_k]
            new_keep = torch.zeros_like(keep)
            new_keep[cutoff] = True
            keep = keep & new_keep

        # top_p: nucleus — keep smallest descending-prob set with cumsum ≥ top_p
        if 0.0 < top_p < 1.0:
            # rank at which cumsum first ≥ top_p (inclusive)
            nucleus_rank = int(torch.searchsorted(cumsum, torch.tensor(top_p, device=cumsum.device)).item()) + 1
            nucleus_rank = max(1, min(nucleus_rank, sorted_idx.numel()))
            nucleus = sorted_idx[:nucleus_rank]
            new_keep = torch.zeros_like(keep)
            new_keep[nucleus] = True
            keep = keep & new_keep

        # Re-sort over the surviving pool for mask_* (mass/rank measured on the
        # POST-top distribution, otherwise mask_p semantics shift with top_p).
        survivor_probs = probs * keep.to(probs.dtype)
        s_total = float(survivor_probs.sum().item())
        if s_total <= 0:
            return probs / probs.sum()
        survivor_norm = survivor_probs / s_total
        sorted_probs2, sorted_idx2 = torch.sort(survivor_norm, descending=True)
        cumsum2 = torch.cumsum(sorted_probs2, dim=0)
        survivors_n = int(keep.sum().item())

        # mask_k: remove the top mask_k highest-prob (within survivors) tokens
        if 0 < mask_k < survivors_n:
            drop = sorted_idx2[:mask_k]
            keep[drop] = False

        # mask_p: remove most-likely tokens summing (cumulative) to ≥ mask_p
        if 0.0 < mask_p < 1.0 and survivors_n > 0:
            drop_rank = int(torch.searchsorted(cumsum2, torch.tensor(mask_p, device=cumsum2.device)).item()) + 1
            drop_rank = min(drop_rank, survivors_n - 1)  # never empty the pool
            if drop_rank > 0:
                drop = sorted_idx2[:drop_rank]
                keep[drop] = False

        filtered = probs * keep.to(probs.dtype)
        total = float(filtered.sum().item())
        if total <= 0:
            return probs / probs.sum()
        return filtered / total

    def _is_acceptable(
        self, original: Score, candidate: Score, cfg: InferenceConfig, step: GenerationStep
    ) -> bool:
        logging.debug(
            f"Checking acceptability: silence_check={cfg.silence_check}, novelty_check={cfg.novelty_check}"
        )

        # Dump request vs. step alignment + per-track/bar note counts.
        try:
            btg = list(step.bars_to_generate)
        except Exception:
            btg = "?"
        logging.debug(
            f"  step: start_bar={step.start_bar} end_bar={step.end_bar} "
            f"track_indices={list(step.track_indices)} "
            f"bars_to_generate={btg}"
        )
        for i, t in enumerate(candidate.tracks):
            counts = [len(b.notes) for b in t.bars]
            logging.debug(
                f"  candidate.tracks[{i}] type={t.track_type} "
                f"instrument={t.instrument} bars={len(t.bars)} "
                f"notes_per_bar={counts}"
            )
        for tp in self._request.tracks:
            logging.debug(
                f"  request.tracks tp.id={tp.id} bars={tp.bars} "
                f"autoreg={tp.autoregressive} ignore={tp.ignore}"
            )

        if cfg.silence_check:
            notes_added = self._note_count(candidate, step)
            logging.debug(f"  Notes added in this step (in generated bars): {notes_added}")
            if notes_added <= 0:
                logging.debug(
                    "  Rejected by silence_check: No new notes in generated bars or notes removed."
                )
                return False
        if cfg.novelty_check:
            if self._too_similar(original, candidate, step, cfg):
                logging.debug("  Rejected by novelty_check: Candidate is too similar to original.")
                return False

        logging.debug("  Accepted: Candidate is acceptable.")
        return True

    def _note_count(self, score: Score, step: GenerationStep) -> int:
        count = 0
        for tp in self._request.tracks:
            # Check if this track is the one being generated in this step

            for bar_idx in tp.bars:
                if bar_idx >= len(score.tracks[tp.id].bars):
                    continue
                bar = score.tracks[tp.id].bars[bar_idx]

                # Only count notes in generated bars for the current step
                if (tp.id, bar_idx) in step.bars_to_generate:
                    for note in bar.notes:
                        if note.pitch >= 0:  # Just count notes with a valid pitch
                            count += 1
        return count

    def _too_similar(self, original: Score, candidate: Score, step, cfg) -> bool:
        """novelty_check: is the candidate too close to the original?

        novelty_threshold 1.0 rejects only an unchanged result (same pitch and
        onset for every note). Below 1.0 the step's generated bars are compared
        as sets of (bar, pitch, onset) notes, so velocity, duration and
        microtiming don't count, and the candidate is rejected when the notes
        they share make up at least that fraction of all their notes.
        """
        threshold = getattr(cfg, "novelty_threshold", 1.0)
        if threshold >= 1.0:
            return self._is_identical(original, candidate)
        return self._similarity(original, candidate, step) >= threshold

    @staticmethod
    def _novelty_message(cfg) -> str:
        threshold = getattr(cfg, "novelty_threshold", 1.0)
        if threshold >= 1.0:
            return "novelty_check (candidate identical to original)"
        return f"novelty_check (candidate at least {threshold:.0%} like the original)"

    @staticmethod
    def _similarity(a: Score, b: Score, step) -> float:
        """Shared (pitch, onset) notes over all notes in the step's generated
        bars (Jaccard, counting repeated notes); 1.0 when both are empty."""
        from collections import Counter

        def notes(score: Score) -> Counter:
            out: Counter = Counter()
            for t, bar in step.bars_to_generate:
                if t < len(score.tracks) and bar < len(score.tracks[t].bars):
                    out.update((bar, n.pitch, n.onset_ticks)
                               for n in score.tracks[t].bars[bar].notes)
            return out

        na, nb = notes(a), notes(b)
        union = sum((na | nb).values())
        return 1.0 if union == 0 else sum((na & nb).values()) / union

    @staticmethod
    def _restore_window(before: Score, after: Score, step) -> Score:
        """Give every track of a decoded step result its full bar count back.

        SessionState splices the bars outside the step window around the
        decoded window. A window that decodes short (the token budget ran out,
        or a partial decode mid-generation) would shift every later bar
        earlier and shrink the track, so the missing window bars are put back
        empty, where the window ends.
        """
        for tb, ta in zip(before.tracks, after.tracks):
            missing = len(tb.bars) - len(ta.bars)
            if missing <= 0:
                continue
            at = max(min(step.end_bar, len(tb.bars)) - missing, 0)
            ta.bars[at:at] = [
                Bar(ts_numerator=b.ts_numerator, ts_denominator=b.ts_denominator,
                    beat_length=b.beat_length)
                for b in tb.bars[at:at + missing]
            ]
        return after

    def _is_identical(self, a: Score, b: Score) -> bool:
        for tp in self._request.tracks:
            for bar_idx in tp.bars:
                ta = (
                    a.tracks[tp.id].bars[bar_idx]
                    if tp.id < len(a.tracks) and bar_idx < len(a.tracks[tp.id].bars)
                    else None
                )
                tb = (
                    b.tracks[tp.id].bars[bar_idx]
                    if tp.id < len(b.tracks) and bar_idx < len(b.tracks[tp.id].bars)
                    else None
                )
                if ta is None and tb is None:
                    continue
                if ta is None or tb is None:
                    return False
                if sorted((n.pitch, n.onset_ticks) for n in ta.notes) != sorted(
                    (n.pitch, n.onset_ticks) for n in tb.notes
                ):
                    return False
        return True

    def _model_max_context(self) -> int:
        """Read the model's positional context length via the ModelBase protocol.

        TorchScript checkpoints are wrapped in TorchScriptAdapter at load time,
        so every model surfaces max_context().
        """
        return self._engine._model.max_context()

    def _build_selection_mask(self) -> _core.SelectionMask:
        n_tracks = len(self._score.tracks)
        n_bars = max((len(t.bars) for t in self._score.tracks), default=0)

        mask = _core.SelectionMask()
        selected = [[False] * n_bars for _ in range(n_tracks)]
        autoregressive = [False] * n_tracks
        ignore = [False] * n_tracks

        for tp in self._request.tracks:
            if tp.id >= n_tracks:
                continue
            for b in tp.bars:
                if b < n_bars:
                    selected[tp.id][b] = True
            autoregressive[tp.id] = tp.autoregressive
            ignore[tp.id] = tp.ignore

        mask.selected = selected
        mask.autoregressive = autoregressive
        mask.ignore = ignore
        return mask

    def _build_constraints(self, step) -> _core.ConstraintGraph:
        graph = _core.ConstraintGraph()
        grammar = _core.GrammarConstraint()

        # `Track` (TrackStart) and `TrackEnd` tokens are NEVER masked here. The
        # grammar FSM and `set_max_tracks(N)` (below) already decide when they
        # are syntactically legal:
        #   - AR, tracks_per_step=1:  model must sample TrackEnd to terminate.
        #   - AR, tracks_per_step>1:  model samples Track between consecutive
        #                             AR tracks and TrackEnd to close each.
        #   - Infill (any track count): model is in FillInStart→…→FillInEnd
        #                             states; Track/TrackEnd are never visited
        #                             during sampling — they live in the prompt
        #                             only.
        # Hard-masking these tokens for `len(request.tracks) <= 1` removed the
        # exit transition that AR needs and was the cause of the "zero legal
        # tokens" crash on single-track generation.

        # Exact bar count enforcement for autoregressive: each track must end
        # with exactly step.end_bar Bar tokens. (Infill is bounded by FillIn
        # block count instead, so leave the grammar unconstrained.)
        if step.is_autoregressive:
            # Window length — SessionState trims the prompt to this many bars
            # and bar_count_ resets per Track, so the AR grammar should expect
            # exactly window_len Bar tokens. Passing the absolute end_bar would
            # force the model to emit (end_bar - window_len) extra Bar tokens
            # of fake notes per generation, scaling cost with the absolute
            # playhead (O(N²) over a session).
            grammar.set_exact_bars(step.end_bar - step.start_bar)
            grammar.set_autoregressive_mode(True)
        # Ignored tracks are omitted from the token sequence by the encoder,
        # so the grammar's track_count_ only ever reaches (n_tracks - n_ignored).
        # Using len(score.tracks) here would force the model to emit extra
        # phantom tracks to satisfy max_tracks.
        ignored_ids = {tp.id for tp in self._request.tracks if tp.ignore}
        grammar.set_max_tracks(len(self._score.tracks) - len(ignored_ids))
        grammar.set_require_notes(True)

        graph.add_constraint(grammar)

        # Global hard cap on simultaneous note onsets — applies to every step
        # regardless of attribute controls. 0 = off.
        hard_limit = int(getattr(self._request.config, "polyphony_hard_limit", 0) or 0)
        if hard_limit > 0:
            graph.add_constraint(_core.PolyphonyConstraint(hard_limit))
        density_limit = int(getattr(self._request.config, "density_hard_limit", 0) or 0)
        if density_limit > 0:
            graph.add_constraint(_core.DensityConstraint(density_limit))

        attr_to_token = {
            "note_density": _core.TokenType.NoteDensity,
            "onset_polyphony": _core.TokenType.OnsetPolyphony,
            "pitch_range": _core.TokenType.PitchRange,
            "key_signature": _core.TokenType.KeySignature,
            "note_duration_dist": _core.TokenType.NoteDurationDist,
            "silence_proportion": _core.TokenType.SilenceProportion,
            "pitch_class_set": _core.TokenType.PitchClassSet,
            "min_note_duration": _core.TokenType.MinNoteDuration,
            "max_note_duration": _core.TokenType.MaxNoteDuration,
            "min_polyphony": _core.TokenType.MinPolyphony,
            "max_polyphony": _core.TokenType.MaxPolyphony,
        }
        # First-class non-attribute controls. These don't flow through the
        # AttributeAnalyzer (no per-bar/per-track computation); they just pin a
        # token to a specific value when generated. Live on tp.controls.
        control_to_token = {
            "time_signature": _core.TokenType.TimeSig,
        }

        full_ar_ids = getattr(self, "_full_ar_ids", set())
        # Ordinal of each generating track in this step (position among the
        # step's track_indices in emit order). Used by per-bar constraints
        # which need to know "are we currently inside the target track?".
        track_ordinal = {tid: i for i, tid in enumerate(step.track_indices)}
        # Analyzer (for bar-level attr_name -> token_type lookup).
        analyzer = self._engine._analyzer

        # Bar attribute name -> TokenType (resolved via analyzer so we don't
        # hardcode the bar-level attribute schema here).
        def _bar_attr_token(name):
            if analyzer is None:
                return None
            obj = analyzer.get(name) if hasattr(analyzer, "get") else None
            if obj is None or getattr(obj, "level", "track") != "bar":
                return None
            tt_name = getattr(obj, "token_type", None)
            return getattr(_core.TokenType, tt_name, None) if tt_name else None

        # Bar control name -> TokenType.
        bar_control_to_token = {
            "time_signature": _core.TokenType.TimeSig,
        }
        for tp in self._request.tracks:
            # Attribute constraints only apply to full-AR tracks; infill and
            # partial-AR pin attributes through the encoded prompt.
            if tp.id not in full_ar_ids:
                continue
            if tp.id in step.track_indices:
                for attr_name, token_type in attr_to_token.items():
                    if attr_name in tp.attributes:
                        val = tp.attributes[attr_name]
                        graph.add_constraint(_core.AttributeValueConstraint(token_type, val))

                # Non-attribute controls: same constraint mechanism, separate
                # source dict to keep the attribute pipeline analyzer-pure.
                controls = getattr(tp, "controls", {}) or {}
                for ctrl_name, token_type in control_to_token.items():
                    if ctrl_name in controls:
                        graph.add_constraint(
                            _core.AttributeValueConstraint(token_type, int(controls[ctrl_name]))
                        )

                # Per-bar overrides — only meaningful for full-AR since
                # infill/partial-AR pin per-bar values via the prompt.
                # Bar-index is RELATIVE to step.start_bar (the grammar
                # resets bar_count_ at each Track token). For full-AR the
                # whole track is generated, so start_bar==0 and absolute
                # == relative.
                t_ord = track_ordinal.get(tp.id, 0)
                start_bar = int(step.start_bar)
                for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
                    rel = int(bar_idx) - start_bar
                    if rel < 0:
                        continue
                    for attr_name, val in (bar_dict or {}).items():
                        tok = _bar_attr_token(attr_name)
                        if tok is None:
                            continue
                        graph.add_constraint(
                            _core.BarAttributeValueConstraint(tok, t_ord, rel, int(val))
                        )
                for bar_idx, bar_dict in (tp.bar_controls or {}).items():
                    rel = int(bar_idx) - start_bar
                    if rel < 0:
                        continue
                    for ctrl_name, val in (bar_dict or {}).items():
                        tok = bar_control_to_token.get(ctrl_name)
                        if tok is None:
                            continue
                        graph.add_constraint(
                            _core.BarAttributeValueConstraint(tok, t_ord, rel, int(val))
                        )

        return graph
