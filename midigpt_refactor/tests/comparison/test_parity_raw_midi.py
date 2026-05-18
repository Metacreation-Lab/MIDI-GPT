import pytest
import json
from .conftest import silence_stdio, pretty_ref, pretty_orig, diff_report, midi_files
import midigpt_refactor._core as _core
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.attributes import AttributeAnalyzer

def _get_test_midis(max_files: int = 5):
    results = []
    # Use refactored reader with high resolution for filtering
    reader = _core.MidiReader(480)
    for mp in midi_files():
        try:
            s = reader.read(str(mp))
            # Require at least 1 bar
            if max(len(t.bars) for t in s.tracks) > 0:
                results.append(str(mp))
                if len(results) >= max_files:
                    break
        except Exception:
            pass
    return results

def _orig_encode_ar_from_midi(orig_enc, midi_path):
    with silence_stdio():
        try:
            pj = orig_enc.midi_to_json(midi_path)
            return orig_enc.json_to_tokens(pj)
        except Exception:
            return None

def test_raw_parity_from_midi(yellow_orig_encoder, yellow_config_text):
    """Bit-perfect parity: orig encode → remap to ref IDs == ref encode of
    the same MIDI. NO normalization, NO header stripping."""
    from midigpt_refactor.compat import build_orig_to_ref_mapping, remap_orig_tokens
    cfg = _core.EncoderConfig.from_json(yellow_config_text)
    analyzer = AttributeAnalyzer.from_config(cfg)
    tokenizer = Tokenizer(cfg, analyzer)
    mapping = build_orig_to_ref_mapping(yellow_orig_encoder, tokenizer._vocab)

    midis = _get_test_midis(max_files=1000)
    if not midis:
        pytest.skip("No suitable test MIDIs found.")

    for mp in midis:
        orig = _orig_encode_ar_from_midi(yellow_orig_encoder, mp)
        if orig is None: continue

        orig_remapped = remap_orig_tokens(orig, mapping)

        reader = _core.MidiReader(12)
        score = reader.read(mp)
        ref = tokenizer.encode(score, compute_attributes=True)

        if orig_remapped != ref:
            msg = f"Bit-perfect parity failed on {mp}\n"
            msg += diff_report("OrigRemapped", orig_remapped, lambda t: pretty_ref(tokenizer._vocab, t),
                               "Ref         ", ref,           lambda t: pretty_ref(tokenizer._vocab, t))
            pytest.fail(msg)
