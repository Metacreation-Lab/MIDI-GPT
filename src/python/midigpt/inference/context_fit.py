"""Which context tracks to leave out when a request doesn't fit the model's window.

A step needs its prompt plus the tokens it will generate (estimated, times a safety margin) to fit. When one does
not, whole context tracks are dropped for the request, in this order:

1. the caller's `context_drop` list, as given;
2. (context "auto") the tracks the model attends to least: attention from the last prompt token, summed over the
   top half of the layers, per token of the track. Measured on GigaMIDI (teacher-forced likelihood of the real
   bars at fixed budgets) this kept the most of the context's value of the policies tried;
3. when the prompt alone is too long to read the model's attention, a role table: how much a track of one role
   (bass, harmony, lead, drum kit, hats, percussion) helps predict a target of another, per token.

Dropping single bars instead of whole tracks, and shrinking the window instead of dropping context, both did
worse, so neither is used here; the session's window step-down is the last resort.
"""

from __future__ import annotations

import statistics

import numpy as np

# ---------------------------------------------------------------- roles

REL = {   # REL[target kind][context role]
    "melodic": {"bass": .9, "harmony": .8, "lead": .6, "kit": .6, "hats": .35, "perc": .35, "other": .3},
    "bass":    {"bass": .6, "harmony": .8, "lead": .5, "kit": .95, "hats": .5, "perc": .5, "other": .3},
    "drums":   {"bass": .8, "harmony": .4, "lead": .3, "kit": .9, "hats": .7, "perc": .6, "other": .3},
}
KIT = (35, 36, 37, 38, 39, 40)
HATS = (42, 44, 46, 49, 51, 52, 53, 55, 57, 59)


def track_role(track, resolution: int) -> str:
    """bass / harmony / lead from register and polyphony; kit / hats / perc from the drum notes."""
    notes = [n for b in track.bars for n in b.notes]
    if not notes:
        return "other"
    if track.track_type == "drum":
        pitches = [n.pitch for n in notes]
        core = sum(p in KIT for p in pitches)
        hats = sum(p in HATS for p in pitches)
        if core >= hats and core > 0.25 * len(notes):
            return "kit"
        return "hats" if hats > 0.5 * len(notes) else "perc"
    grid = max(1, resolution // 4)
    onsets: dict = {}
    for bi, b in enumerate(track.bars):
        for n in b.notes:
            k = (bi, round(n.onset_ticks / grid))
            onsets[k] = onsets.get(k, 0) + 1
    poly = float(np.mean(list(onsets.values())))
    if np.mean([n.pitch for n in notes]) < 50 and poly < 1.3:
        return "bass"
    return "harmony" if poly >= 1.8 else "lead"


def _kind(role: str) -> str:
    return "drums" if role in ("kit", "hats", "perc") else ("bass" if role == "bass" else "melodic")


def rank_by_roles(score, targets, candidates, costs) -> list[int]:
    """Candidates in drop order: lowest relevance to the targets (best target) per token first."""
    res = score.resolution
    roles = {t: track_role(score.tracks[t], res) for t in set(targets) | set(candidates)}
    family = {t: (score.tracks[t].instrument // 8 if score.tracks[t].track_type != "drum" else -1) for t in roles}

    def relevance(c):
        return max(REL[_kind(roles[t])][roles[c]] + (0.1 if family[t] == family[c] else 0.0) for t in targets)

    return sorted(candidates, key=lambda c: (relevance(c) / max(1, costs.get(c, 1)), c))


# ---------------------------------------------------------------- attention

def attention_from_last(model, tokens, spans=()) -> np.ndarray | None:
    """Attention from the last prompt token to every prompt token, heads averaged, summed over the top half of the
    layers. None when the model doesn't expose its blocks (TorchScript)."""
    import torch

    tr = getattr(model, "transformer", None)
    if tr is None or not hasattr(tr, "h"):
        return None
    dev = next(model.parameters()).device
    T = len(tokens)
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.long, device=dev)
        x = tr.wte(ids) + tr.wpe(torch.arange(T, device=dev)[None])
        km = None
        if spans:
            km = torch.ones(T, dtype=torch.bool, device=dev)
            for a, b in spans:
                km[a:b] = False
        L = len(tr.h)
        acc = torch.zeros(T, device=dev)
        for i, blk in enumerate(tr.h):
            want = i >= L // 2
            x, _, w = blk(x, past_kv=None, return_attn_weights=want, key_mask=km)
            if want:
                acc += w[0, :, -1, :].float().mean(0)
    return acc.cpu().numpy()


def rank_by_attention(attention: np.ndarray, track_of_token, candidates, costs) -> list[int]:
    """Candidates in drop order: least attention per token first."""
    tot = np.array(track_of_token)
    mass = {c: float(attention[tot == c].sum()) / max(1, costs.get(c, 1)) for c in candidates}
    return sorted(candidates, key=lambda c: (mass[c], c))


# ---------------------------------------------------------------- measuring a step

def track_of_token(tokens, vocab, order) -> list[int]:
    """The track (index into `order`, the encoded track order) each prompt token belongs to; -1 for the header
    and anything after the tracks."""
    import midigpt._core as _core

    out, k, cur = [], -1, -1
    for x in tokens:
        ty = vocab.get_type(x)
        if ty == _core.TokenType.Track:
            k += 1
            cur = order[k] if k < len(order) else -1
        elif ty in (_core.TokenType.FillInStart, _core.TokenType.PieceEnd):
            cur = -1
        out.append(cur)
        if ty == _core.TokenType.TrackEnd:
            cur = -1
    return out


def bar_tokens(tokens, vocab, owners) -> dict:
    """{(track, bar index within the window): tokens} for the bars in the prompt."""
    import midigpt._core as _core

    out: dict = {}
    bar, prev = -1, None
    for x, t in zip(tokens, owners):
        if t != prev:
            bar, prev = -1, t
        if t < 0:
            continue
        if vocab.get_type(x) == _core.TokenType.Bar:
            bar += 1
        if bar >= 0:
            out[(t, bar)] = out.get((t, bar), 0) + 1
    return out


def estimate_generated(score, step, per_bar: dict) -> int:
    """Tokens the step will generate: per target bar, its current notes at the track's tokens per note, or the
    median of the track's context bars in the window, or a prior; plus the bar's structure."""
    est = 0
    for t, b in step.bars_to_generate:
        track = score.tracks[t]
        drum = track.track_type == "drum"
        ctx = [n for (tt, lb), n in per_bar.items() if tt == t and n > 4]
        notes_ctx = sum(len(track.bars[step.start_bar + lb].notes) for (tt, lb), n in per_bar.items()
                        if tt == t and n > 4 and step.start_bar + lb < len(track.bars))
        per_note = (sum(ctx) - 4 * len(ctx)) / notes_ctx if notes_ctx else (2.5 if drum else 3.5)
        now = len(track.bars[b].notes) if b < len(track.bars) else 0
        if now:
            est += int(now * per_note) + 4
        elif ctx:
            est += int(statistics.median(ctx))
        else:
            est += 60 if drum else 40
    return est
