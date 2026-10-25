"""Pieces and requests: picks GigaMIDI test pieces of each kind the `piece` axis names, cuts a segment, and turns
a case into (score, request dict) for one model, the way a client would send it.
"""

from __future__ import annotations

import copy
import glob
import os
import random

from midigpt._types import Score, Track

MODEL_RES = 12  # TimeAbsolutePos steps per beat, every current checkpoint


# ------------------------------------------------------------------ pieces

def _meter(bar):
    return bar.ts_numerator, bar.ts_denominator


def _kinds(score: Score) -> set[str]:
    """Every piece kind a score can stand for."""
    nb = min(len(t.bars) for t in score.tracks)
    meters = {_meter(t.bars[b]) for t in score.tracks for b in range(nb)}
    drums = any(t.track_type == "drum" for t in score.tracks)
    notes = sum(len(b.notes) for t in score.tracks for b in t.bars[:nb])
    per_bar_track = notes / max(1, nb * len(score.tracks))
    out = set()
    if len(meters) == 1:
        m = next(iter(meters))
        if m == (4, 4) and drums and len(score.tracks) >= 4:
            out.add("drums44")
        if m == (4, 4) and not drums and len(score.tracks) >= 4:
            out.add("nodrums44")
        if m == (3, 4) and drums:
            out.add("threefour")
        if m == (6, 8):
            out.add("sixeight")
        if per_bar_track < 3 and len(score.tracks) >= 4:
            out.add("sparse")
        if len(score.tracks) >= 10 and drums:
            out.add("wide")
    return out


def load_pieces(data_dir: str, per_kind: int = 2, seed: int = 0, max_scan: int = 20000) -> dict[str, list]:
    """{kind: [(piece id, Score), ...]} from the GigaMIDI parquet shards in data_dir (or .mid files)."""
    want = ["drums44", "nodrums44", "threefour", "sixeight", "sparse", "wide"]
    out = {k: [] for k in want}
    rng = random.Random(seed)

    def offer(pid, score):
        if not score.tracks or min(len(t.bars) for t in score.tracks) < 36:
            return
        for k in sorted(_kinds(score)):
            if len(out[k]) < per_kind:
                out[k].append((pid, score))
                return

    mids = sorted(glob.glob(os.path.join(data_dir, "**", "*.mid*"), recursive=True))
    if mids:
        rng.shuffle(mids)
        for p in mids[:max_scan]:
            try:
                offer(os.path.basename(p), Score.from_midi(p))
            except Exception:
                continue
            if all(len(v) >= per_kind for v in out.values()):
                break
        return out

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["num_tracks", "total_notes"]).to_pylist()
        rows += [(f, i) for i, r in enumerate(t) if 3 <= (r["num_tracks"] or 0) <= 16 and (r["total_notes"] or 0) >= 200]
    rng.shuffle(rows)
    tables = {}
    for f, i in rows[:max_scan]:
        if all(len(v) >= per_kind for v in out.values()):
            break
        if f not in tables:
            tables[f] = pq.read_table(f, columns=["music", "md5"])
        rec = tables[f].slice(i, 1).to_pylist()[0]
        try:
            offer(rec["md5"][:10], Score.from_bytes(rec["music"]))
        except Exception:
            continue
    return out


def segment(score: Score, length: int, start: int) -> Score:
    tracks = [Track(bars=copy.deepcopy(t.bars[start:start + length]), instrument=t.instrument,
                    track_type=t.track_type) for t in score.tracks]
    return Score(tracks=tracks, resolution=score.resolution, tempo=score.tempo)


def pick_segment(score: Score, length: int, rng: random.Random) -> Score:
    """A `length`-bar window where most tracks play (the busiest of a few tries)."""
    nb = min(len(t.bars) for t in score.tracks)
    best, best_n = 0, -1
    for _ in range(8):
        s = rng.randint(0, max(0, nb - length))
        n = sum(1 for t in score.tracks for b in t.bars[s:s + length] if b.notes)
        if n > best_n:
            best, best_n = s, n
    return segment(score, length, best)


# ------------------------------------------------------------------ requests

def _playing(score, t):
    return sum(1 for b in score.tracks[t].bars if b.notes)


def _target_bars(case, nb):
    shape = case["bars"]
    if case["mode"] == "ar_full":
        return list(range(nb))
    if case["mode"] in ("ar_partial",):
        return [nb - 1] if shape == "one" else list(range(nb // 2, nb))
    if shape == "one":
        return [nb // 2]
    if shape == "contiguous":
        return [nb // 2 - 1, nb // 2]
    if shape == "separate":
        return list(range(1, nb, 2))[: max(2, nb // 4)]
    return list(range(nb))


def _bar_positions(bar) -> list[dict]:
    """A hard rhythm schedule for this bar: quarter notes (dotted quarters in 6/8), from 0."""
    beats = bar.ts_numerator * 4 / bar.ts_denominator
    step = MODEL_RES * (3 if (bar.ts_numerator, bar.ts_denominator) == (6, 8) else 2) // 2
    return [{"pos": p, "polyphony": 1} for p in range(0, int(beats * MODEL_RES), step)]


def _control(name, bar):
    if name.startswith("remix_"):
        _, mode, amt = name.split("_")
        return {"remix": {"mode": mode, "amount": {"1": 1.0, "half": 0.5, "0": 0.0}[amt]}}
    return {
        "none": {},
        "rhythm_positions": {"rhythm_mask": {"positions": _bar_positions(bar)}},
        "rhythm_grid": {"rhythm_mask": {"grid": {"unit": "eighth", "strength": 1.0}}},
        "pitch_scale": {"pitch_mask": {"scale": "dorian", "root": 2}},
        "pitch_classes": {"pitch_mask": {"pitch_classes": [0, 4, 7]}},
        "pitch_list": {"pitch_mask": {"pitches": [36, 38, 42, 60, 62, 64, 67]}},
        "pitch_shape": {"pitch_mask": {"shape": {"type": "uniform", "min": 48, "max": 72}}},
        "scale_and_grid": {"pitch_mask": {"scale": "major_pentatonic", "root": 7},
                           "rhythm_mask": {"grid": {"unit": "sixteenth", "strength": 1.0}}},
    }[name]


def _attributes(case, engine, score, tid, bars, rng):
    an = engine._analyzer
    sizes, levels, types = an.attribute_sizes(), an.attribute_levels(), an.attribute_track_types()
    drum = score.tracks[tid].track_type == "drum"
    kind = case["attributes"]
    track, per_bar = {}, {}

    def ok(name):
        tt = types.get(name, "both")
        return tt == "both" or (tt == "drum") == drum

    def val(name, i):
        sz = max(1, sizes.get(name, 1))
        if kind == "extremes":
            return 0 if (i + tid) % 2 == 0 else sz - 1
        return rng.randrange(sz)

    for i, name in enumerate(sorted(sizes)):
        if not ok(name):
            continue
        lvl = levels.get(name, "track")
        if lvl == "track" and kind in ("track", "both", "extremes"):
            track[name] = val(name, i)
        if lvl == "bar" and kind in ("bar", "both", "extremes"):
            for b in bars:
                per_bar.setdefault(b, {})[name] = val(name, i + b)
    return track, per_bar


def build(case: dict, engine, pieces: dict, seed: int):
    """(score, request dict, info) for a case, or raises LookupError when no piece fits it."""
    rng = random.Random(seed)
    pool = pieces.get(case["piece"]) or []
    if not pool:
        raise LookupError(f"no {case['piece']} piece")
    pid, full = pool[seed % len(pool)]
    score = pick_segment(full, case["length"], rng)
    nb = case["length"]
    n = len(score.tracks)

    melodic = sorted((t for t in range(n) if score.tracks[t].track_type != "drum" and _playing(score, t)),
                     key=lambda t: -_playing(score, t))
    drums = [t for t in range(n) if score.tracks[t].track_type == "drum" and _playing(score, t)]
    k = case["targets"]
    if case["target_kind"] == "drum":
        targets = drums[:k]
    elif case["target_kind"] == "both":
        targets = drums[:1] + melodic[: k - 1]
    else:
        targets = melodic[:k]
    if not targets or (case["target_kind"] != "melodic" and not drums):
        raise LookupError("no track of the kind asked for plays in the segment")

    rest = [t for t in range(n) if t not in targets]
    ignored = [] if case["ignore"] == "none" else rest[:1] if case["ignore"] == "one" else rest[: len(rest) // 2]
    context = [t for t in rest if t not in ignored]

    bars = _target_bars(case, nb)
    tracks = []
    for t in range(n):
        tp = {"id": t, "bars": [], "autoregressive": False, "ignore": t in ignored, "mask_bars": [],
              "attributes": {}, "bar_attributes": {}, "controls": {}}
        if t in targets:
            ar = case["mode"] in ("ar_full", "ar_partial") or (case["mode"] == "mixed" and t == targets[-1])
            if case["mode"] == "mixed":
                tp["bars"] = list(range(nb // 2, nb)) if ar else [nb // 2 - 1]
            else:
                tp["bars"] = list(bars)
            tp["autoregressive"] = ar
            tp["controls"] = _control(case["control"], score.tracks[t].bars[tp["bars"][0]])
            tp["attributes"], tp["bar_attributes"] = _attributes(case, engine, score, t, tp["bars"], rng)
        elif t in context and case["mask_bars"] == "some" and t == context[0]:
            tp["mask_bars"] = [b for b in (1, nb - 2) if 0 <= b < nb]
        tracks.append(tp)

    cfg = {
        "seed": seed,
        "model_dim": case["model_dim"], "bars_per_step": case["bars_per_step"],
        "tracks_per_step": case["tracks_per_step"], "context": case["context"],
        "mask_mode": case["mask_mode"], "infill_probability": case["prob"],
        "num_candidates": case["takes"], "batch_attempts": case["batch_attempts"], "shuffle": case["shuffle"],
        "polyphony_hard_limit": 2 if case["limits"] in ("poly", "both") else 0,
        "density_hard_limit": 8 if case["limits"] in ("density", "both") else 0,
    }
    if case["context"] == "listed" and context:
        cfg["context_drop"] = [context[-1]]
        cfg["context_keep"] = context[:1]
    cfg.update({"default": {}, "top_p": {"top_p": 0.9}, "top_k": {"top_k": 20}, "mask_p": {"mask_p": 0.3},
                "hot": {"temperature": 1.3, "temperature_escalation": 1.2}, "cold": {"temperature": 0.6}}[case["sampling"]])
    cfg.update({"default": {}, "off": {"novelty_check": False, "silence_check": False},
                "one_attempt": {"max_attempts": 1}}[case["checks"]])

    controls = {}
    pc = case["piece_controls"]
    if pc == "genre":
        g = getattr(engine._tokenizer._vocab.config(), "genre_grouping", None)
        controls["genre"] = g.decode(seed % g.num_genres()) if g is not None else "rock"
    if pc in ("velocity_off", "all_off"):
        controls["velocity"] = False
    if pc in ("microtiming_off", "all_off"):
        controls["microtiming"] = False

    req = {"tracks": tracks, "config": cfg, "controls": controls}
    _probe(case.get("probe"), req, score)
    info = {"piece": pid, "targets": targets, "ignored": ignored, "context": context}
    return score, req, info


def _probe(name, req, score):
    """Malformed payloads for the PROBES that need more than axis values."""
    if not name:
        return
    tgt = next(tp for tp in req["tracks"] if tp["bars"])
    nb = len(score.tracks[0].bars)
    if name == "rhythm_not_from_zero":
        tgt["controls"] = {"rhythm_mask": {"positions": [{"pos": 6}, {"pos": 12}]}}
        req["config"]["tracks_per_step"] = 1
    elif name == "pitch_mask_empty":
        tgt["controls"] = {"pitch_mask": {"pitches": []}}
    elif name == "remix_and_rhythm":
        tgt["controls"] = {"remix": {"amount": 0.5}, "rhythm_mask": {"positions": [{"pos": 0}]}}
        req["config"]["tracks_per_step"] = 1
    elif name == "attribute_out_of_range":
        tgt["attributes"] = {"max_note_duration": 10 ** 6}
    elif name == "bar_attribute_off_target":
        off = next(b for b in range(nb) if b not in tgt["bars"]) if len(tgt["bars"]) < nb else None
        if off is not None:
            tgt["bar_attributes"] = {off: {"bar_note_density": 0}}
    elif name == "bar_out_of_range":
        tgt["bars"] = tgt["bars"] + [nb + 3]
    elif name == "unknown_control":
        tgt["controls"] = {"swing": 0.5}
    elif name == "remix_amount_over_one":
        tgt["controls"] = {"remix": {"amount": 1.5}}
        req["config"]["tracks_per_step"] = 1
    elif name == "unknown_genre":
        req["controls"] = {"genre": "no_such_genre"}

