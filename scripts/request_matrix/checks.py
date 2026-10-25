"""What must hold for a generation that came back: each check returns findings, (severity, check, detail), where
severity is "violation" (a hard guarantee broken) or "warning" (worth a look, not necessarily wrong).

Times are compared on the model's grid (MODEL_RES steps per beat), floored as the tokenizer does: the reply
comes back at another resolution and context bars go through the tokenizer, so exact ticks (or an onset a tick
before a grid line) would differ for reasons that aren't bugs.
"""

from __future__ import annotations

from collections import Counter

from build import MODEL_RES

from midigpt.inference.config import resolve_pitch_allow_set


def _q(ticks, res):
    return int(ticks * MODEL_RES // res)


def _beats(bar):
    return bar.ts_numerator * 4 / bar.ts_denominator


def _onsets(bar, res):
    return Counter(_q(n.onset_ticks, res) for n in bar.notes)


def _notes(bar, res):
    return Counter((n.pitch, _q(n.onset_ticks, res)) for n in bar.notes)


def check(case, req, score_in, out, sess, streamed) -> list[tuple[str, str, str]]:
    f = []
    masked = {(tp["id"], b) for tp in req["tracks"] for b in tp.get("mask_bars", [])}
    rin, rout = score_in.resolution, out.resolution
    if len(out.tracks) != len(score_in.tracks):
        return [("violation", "track_count", f"{len(score_in.tracks)} in, {len(out.tracks)} out")]

    gen = {int(t): set(b) for t, b in (sess.generated_bars or {}).items()}
    targets = {tp["id"]: tp for tp in req["tracks"] if tp["bars"] or tp["autoregressive"]}
    for t, tp in targets.items():
        asked = set(tp["bars"]) if tp["bars"] else set(range(len(score_in.tracks[t].bars)))
        got = gen.get(t, set())
        if not got <= asked:
            f.append(("violation", "generated_bars", f"track {t}: {sorted(got - asked)} not asked for"))
        if case["prob"] >= 1.0 and not tp["autoregressive"] and got != asked:
            f.append(("violation", "generated_bars", f"track {t}: asked {sorted(asked)}, got {sorted(got)}"))
    for t in gen:
        if t not in targets:
            f.append(("violation", "generated_bars", f"track {t} generated but not a target"))

    for t, (tin, tout) in enumerate(zip(score_in.tracks, out.tracks, strict=True)):
        if len(tout.bars) != len(tin.bars):
            f.append(("violation", "bar_count", f"track {t}: {len(tin.bars)} in, {len(tout.bars)} out"))
            continue
        for b, (bin_, bout) in enumerate(zip(tin.bars, tout.bars, strict=True)):
            written = b in gen.get(t, set())
            if not written:
                if _notes(bin_, rin) != _notes(bout, rout):
                    if (t, b) in masked:
                        f.append(("warning", "masked_bar_changed", f"track {t} bar {b}: a masked context bar came back changed"))
                    else:
                        f.append(("violation", "context_changed", f"track {t} bar {b}: notes changed outside the generated bars"))
                continue
            length = round(_beats(bin_) * rout)
            for n in bout.notes:
                if not (0 <= n.onset_ticks < length):
                    f.append(("violation", "onset_outside_bar", f"track {t} bar {b}: onset {n.onset_ticks} of {length}"))
                    break
                if n.duration_ticks <= 0 or not (0 <= n.pitch <= 127) or not (0 <= n.velocity <= 127):
                    f.append(("violation", "note_values", f"track {t} bar {b}: {n}"))
                    break

    for t, tp in targets.items():
        ctl = tp["controls"]
        bars = sorted(gen.get(t, set()))
        tin, tout = score_in.tracks[t], out.tracks[t]
        pm = ctl.get("pitch_mask")
        if pm:
            allow = resolve_pitch_allow_set(pm)
            shape = pm.get("shape") or {}
            lo, hi = (shape.get("min", 0), shape.get("max", 127)) if shape.get("type") == "uniform" else (0, 127)
            bad = [n.pitch for b in bars for n in tout.bars[b].notes
                   if (allow is not None and n.pitch not in allow) or not (lo <= n.pitch <= hi)]
            if bad:
                f.append(("violation", "pitch_mask", f"track {t}: pitches {sorted(set(bad))[:8]} outside the mask"))
        rm = ctl.get("rhythm_mask") or {}
        if "positions" in rm:
            want = Counter({p["pos"]: p.get("polyphony", 1) for p in rm["positions"]})
            for b in bars:
                got = _onsets(tout.bars[b], rout)
                if got and set(got) != set(want):
                    f.append(("violation", "rhythm_positions", f"track {t} bar {b}: onsets {sorted(got)} != {sorted(want)}"))
                    break
        if "grid" in rm and float(rm["grid"].get("strength", 1.0)) >= 1.0:
            step = {"eighth": MODEL_RES // 2, "sixteenth": MODEL_RES // 4}.get(rm["grid"]["unit"], 1)
            off = [o for b in bars for o in _onsets(tout.bars[b], rout) if o % step]
            if off:
                f.append(("violation", "rhythm_grid", f"track {t}: onsets {sorted(set(off))[:8]} off the {rm['grid']['unit']} grid"))
        rx = ctl.get("remix")
        if rx:
            for b in bars:
                ref, got = tin.bars[b], tout.bars[b]
                if set(_onsets(ref, rin)) != set(_onsets(got, rout)):
                    f.append(("violation", "remix_onsets", f"track {t} bar {b}: onsets {sorted(_onsets(got, rout))} "
                                                           f"!= reference {sorted(_onsets(ref, rin))}"))
                    break
                if float(rx["amount"]) == 0.0 and _notes(ref, rin) != _notes(got, rout):
                    f.append(("violation", "remix_amount_0", f"track {t} bar {b}: amount 0 changed pitches"))
                    break

        cfg = req["config"]
        if cfg.get("polyphony_hard_limit"):
            worst = max((max(_onsets(tout.bars[b], rout).values(), default=0) for b in bars), default=0)
            if worst > cfg["polyphony_hard_limit"]:
                f.append(("warning", "polyphony_limit", f"track {t}: {worst} notes at one onset, limit {cfg['polyphony_hard_limit']}"))
        if cfg.get("density_hard_limit"):
            worst = max((len(tout.bars[b].notes) for b in bars), default=0)
            if worst > cfg["density_hard_limit"]:
                f.append(("warning", "density_limit", f"track {t}: {worst} notes in a bar, limit {cfg['density_hard_limit']}"))
        if cfg.get("silence_check", True) and bars and not any(tout.bars[b].notes for b in bars):
            f.append(("warning", "silent", f"track {t}: every generated bar is empty with silence_check on"))

    if req["controls"].get("velocity") is False:
        vels = {n.velocity for t in targets for b in gen.get(t, ()) for n in out.tracks[t].bars[b].notes}
        if len(vels) > 1:
            f.append(("warning", "velocity_off", f"{len(vels)} distinct velocities with velocity off"))
    if getattr(sess, "truncated", False):
        f.append(("warning", "truncated", f"ran into the context limit ({sess.gen_count} tokens)"))
    n_bars = sum(len(b) for b in gen.values()) or 1
    if sess.gen_count / n_bars > 600:
        f.append(("violation", "runaway", f"{sess.gen_count} tokens for {n_bars} bars"))
    if streamed is not None:
        total = sum(len(out.tracks[t].bars[b].notes) for t in gen for b in gen[t])
        if total and not streamed:
            f.append(("warning", "stream_empty", f"{total} notes written, none streamed"))
    return f


def same_result(a, b) -> bool:
    return [[_notes(bar, a.resolution) for bar in t.bars] for t in a.tracks] == \
           [[_notes(bar, b.resolution) for bar in t.bars] for t in b.tracks]
