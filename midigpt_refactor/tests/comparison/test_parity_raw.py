import pytest
import json
from .conftest import silence_stdio, pretty_ref, pretty_orig, diff_report, midi_files
import midigpt_refactor._core as _core

def _get_test_midis(min_bars: int = 4, max_files: int = 5):
    results = []
    with silence_stdio():
        for mp in midi_files():
            try:
                s = _core.MidiReader().read(str(mp))
                if max(len(t.bars) for t in s.tracks) >= min_bars:
                    results.append(str(mp))
                    if len(results) >= max_files:
                        break
            except Exception:
                pass
    return results

def _orig_encode_ar(orig_enc, midi_path, n_bars):
    with silence_stdio():
        try:
            pj = orig_enc.midi_to_json(midi_path)
        except Exception:
            return None
    d = json.loads(pj)
    for t in d.get("tracks", []):
        t["bars"] = t.get("bars", [])[:n_bars]
    pj_trimmed = json.dumps(d)
    with silence_stdio():
        try:
            return orig_enc.json_to_tokens(pj_trimmed)
        except Exception:
            return None

def test_raw_parity_autoregressive(yellow_orig_encoder, yellow_ref_components):
    """orig encode → remap to ref IDs → ref decode → ref re-encode.
    Round-tripped tokens must equal the remapped orig tokens."""
    from midigpt_refactor.compat import build_orig_to_ref_mapping, remap_orig_tokens
    cfg, vocab, ref_enc, ref_dec = yellow_ref_components
    mapping = build_orig_to_ref_mapping(yellow_orig_encoder, vocab)
    midis = _get_test_midis(4, max_files=1000)
    if not midis:
        pytest.skip("no test MIDI available")

    compared = 0
    failures = []
    for mp in midis:
        orig = _orig_encode_ar(yellow_orig_encoder, mp, 4)
        if orig is None:
            continue

        orig_remapped = remap_orig_tokens(orig, mapping)

        with silence_stdio():
            score = ref_dec.decode(orig_remapped)

        for t in score.tracks:
            t.attributes.clear()

        ref = ref_enc.encode(score)

        if orig_remapped != ref:
            orig = orig_remapped  # for diff reporting below
            msg = f"Parity failed on {mp}\n"
            msg += f"Orig len: {len(orig)}, Ref len: {len(ref)}\n"
            for i in range(min(len(orig), len(ref), 50)):
                o, r = orig[i], ref[i]
                if o != r:
                    msg += f"Mismatch at index {i}: OrigRemapped={o} ({pretty_ref(vocab, o)}), Ref={r} ({pretty_ref(vocab, r)})\n"
                    break
            failures.append(msg)
        compared += 1

    if failures:
        pytest.fail("\n".join(failures))
