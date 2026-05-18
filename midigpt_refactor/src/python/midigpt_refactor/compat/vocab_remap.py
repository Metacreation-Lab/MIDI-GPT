"""Vocabulary remapping between original midigpt and midigpt_refactor.

Both vocabularies cover the same set of logical tokens (PieceStart,
NumBars, Track, Instrument, TimeSig, NoteOnset, NoteDuration, VelocityLevel,
TimeAbsolutePos, NoteDensity, MinPolyphony, MaxPolyphony, MinNoteDuration,
MaxNoteDuration, Bar, BarEnd, TrackEnd, FillIn*, …) but assign them different
integer IDs and use different value-label conventions (orig uses semantic
values like "1/2" or "127", ref uses dense indices 0..N-1).

The remap is built by aligning each type's token range positionally:
orig's j-th token of type T → ref's j-th token of type T. Both vocabs have
identical counts per type (verified by `assert_compatible`), making the
mapping bijective.

Used to load original checkpoint weights into refactored model:
    mapping = build_orig_to_ref_mapping(orig_enc, ref_vocab)
    new_wte = remap_embedding_weight(orig_wte, mapping, ref_vocab.size())
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional


# orig pretty-name → ref TokenType name. Only names that appear in
# midigpt.ElVelocityDurationPolyphony*Encoder.pretty() output.
_ORIG_TO_REF_TYPE = {
    "TOKEN_PIECE_START":       "PieceStart",
    "TOKEN_PIECE_END":         "PieceEnd",
    "TOKEN_NUM_BARS":          "NumBars",
    "TOKEN_BAR":               "Bar",
    "TOKEN_BAR_END":           "BarEnd",
    "TOKEN_TRACK":             "Track",
    "TOKEN_TRACK_END":         "TrackEnd",
    "TOKEN_INSTRUMENT":        "Instrument",
    "TOKEN_TIME_SIGNATURE":    "TimeSig",
    "TOKEN_TIME_ABSOLUTE_POS": "TimeAbsolutePos",
    "TOKEN_NOTE_ONSET":        "NoteOnset",
    "TOKEN_NOTE_DURATION":     "NoteDuration",
    "TOKEN_VELOCITY_LEVEL":    "VelocityLevel",
    "TOKEN_FILL_IN_PLACEHOLDER": "FillInPlaceholder",
    "TOKEN_FILL_IN_START":     "FillInStart",
    "TOKEN_FILL_IN_END":       "FillInEnd",
    "TOKEN_MASK_BAR":          "MaskBar",
    "TOKEN_DENSITY_LEVEL":     "NoteDensity",
    "TOKEN_POLYPHONY_LEVEL":   "OnsetPolyphony",
    "TOKEN_MIN_POLYPHONY":     "MinPolyphony",
    "TOKEN_MAX_POLYPHONY":     "MaxPolyphony",
    "TOKEN_MIN_NOTE_DURATION": "MinNoteDuration",
    "TOKEN_MAX_NOTE_DURATION": "MaxNoteDuration",
    "TOKEN_PITCH_RANGE":       "PitchRange",
    "TOKEN_KEY_SIGNATURE":     "KeySignature",
    "TOKEN_TENSION":           "Tension",
    "TOKEN_SILENCE_PROPORTION": "SilenceProportion",
    "TOKEN_PITCH_CLASS_SET":   "PitchClassSet",
    "TOKEN_NOTE_DURATION_DIST": "NoteDurationDist",
}


def _group_orig_by_type(orig_enc) -> Dict[str, List[int]]:
    """Return {orig_type_name: [token_id_in_order]} preserving orig ID order."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(orig_enc.vocab_size()):
        p = orig_enc.pretty(i)
        name = p.split(" = ", 1)[0] if " = " in p else p
        groups[name].append(i)
    return dict(groups)


def _group_ref_by_type(ref_vocab) -> Dict[str, List[int]]:
    """Return {ref_type_name: [token_id_in_order]} preserving ref ID order."""
    import midigpt_refactor._core as _core  # noqa: F401  — for enum import
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(ref_vocab.size()):
        try:
            tt, _ = ref_vocab.decode(i)
        except Exception:
            continue
        groups[str(tt).split(".")[-1]].append(i)
    return dict(groups)


def build_orig_to_ref_mapping(orig_enc, ref_vocab, *, strict: bool = True) -> List[int]:
    """Build a length-`orig_vocab_size` list where mapping[orig_id] = ref_id.

    Tokens that have no counterpart in ref (e.g. orig types missing from
    ref's vocabulary) map to -1.

    With strict=True, raises ValueError on a per-type count mismatch.
    With strict=False, the overlap is used (positional alignment within the
    shared prefix).
    """
    orig_groups = _group_orig_by_type(orig_enc)
    ref_groups = _group_ref_by_type(ref_vocab)

    mapping = [-1] * orig_enc.vocab_size()
    mismatches = []
    unmapped_types = []

    for orig_name, orig_ids in orig_groups.items():
        ref_name = _ORIG_TO_REF_TYPE.get(orig_name)
        if ref_name is None or ref_name not in ref_groups:
            unmapped_types.append(orig_name)
            continue

        ref_ids = ref_groups[ref_name]
        if len(orig_ids) != len(ref_ids):
            mismatches.append((orig_name, len(orig_ids), ref_name, len(ref_ids)))
            if strict:
                continue

        n = min(len(orig_ids), len(ref_ids))
        for j in range(n):
            mapping[orig_ids[j]] = ref_ids[j]

    if strict and mismatches:
        details = "; ".join(
            f"{on}({oc}) ≠ {rn}({rc})" for on, oc, rn, rc in mismatches
        )
        raise ValueError(f"Vocabulary type-count mismatches: {details}")

    return mapping


def remap_orig_tokens(orig_tokens, mapping) -> List[int]:
    """Translate a sequence of orig token IDs to ref token IDs.

    Tokens with no mapping (-1) are dropped. Use this to feed a sequence
    produced by the original encoder/model into ref-side decode/inspection.
    """
    return [mapping[t] for t in orig_tokens if 0 <= t < len(mapping) and mapping[t] >= 0]


def remap_embedding_weight(orig_weight, mapping, ref_vocab_size: int,
                            fill: Optional[float] = None):
    """Reorder rows of an embedding weight (orig_vocab_size × dim) to match
    the ref vocab's token-ID layout.

    Args:
        orig_weight: torch.Tensor of shape (orig_vocab_size, dim) or a
                     numpy array — anything that supports `.shape` and
                     row indexing.
        mapping:    list of length orig_vocab_size from build_orig_to_ref_mapping.
        ref_vocab_size: rows in the output.
        fill: value for ref rows that have no orig source. None → zero.

    Returns the same type/dtype/device as `orig_weight`.

    Only rows with a valid mapping[i] are copied; unmapped ref rows (e.g.
    ref-only token types) are filled with `fill` (default 0). Embedding for
    those rows must then be initialized by the caller (random init or
    fine-tuning).
    """
    import numpy as np
    is_torch = hasattr(orig_weight, "detach")

    if is_torch:
        import torch
        dim = orig_weight.shape[1]
        new = torch.zeros(
            (ref_vocab_size, dim),
            dtype=orig_weight.dtype, device=orig_weight.device,
        )
        if fill is not None:
            new.fill_(fill)
        for orig_id, ref_id in enumerate(mapping):
            if ref_id >= 0:
                new[ref_id] = orig_weight[orig_id]
        return new

    arr = np.asarray(orig_weight)
    dim = arr.shape[1]
    new = np.zeros((ref_vocab_size, dim), dtype=arr.dtype)
    if fill is not None:
        new[:] = fill
    for orig_id, ref_id in enumerate(mapping):
        if ref_id >= 0:
            new[ref_id] = arr[orig_id]
    return new
