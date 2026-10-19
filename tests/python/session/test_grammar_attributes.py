"""The generation grammar accepts every token of an encoded piece, attribute tokens included: prism writes
track-level key / silence and bar-level polyphony before TimeSig, which the grammar used to mask."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_bar, make_note, melodic_track

import midigpt._core as _core
from midigpt._types import Score, Track
from midigpt.attributes import AttributeAnalyzer
from midigpt.tokenizer import Tokenizer

MODELS = Path(__file__).resolve().parents[3] / "models"


def _piece():
    drums = Track(bars=[make_bar([make_note(pitch=36, onset=i * 12, dur=1) for i in range(4)]) for _ in range(4)],
                  instrument=0, track_type="drum")
    return Score(tracks=[melodic_track(n_bars=4, notes_per_bar=4), drums], resolution=12)


@pytest.mark.parametrize("config", ["prism_encoder.json", "yellow_config.json"])
def test_grammar_accepts_every_encoded_token(config):
    cfg = _core.EncoderConfig.from_json((MODELS / config).read_text())
    tok = Tokenizer(cfg, AttributeAnalyzer.from_config(cfg))
    vocab = tok._vocab
    tokens = tok.encode(_piece())
    names = [str(vocab.decode(t)[0]).split(".")[-1] for t in tokens]
    if config.startswith("prism"):                       # the case this guards: attribute tokens are there
        assert "BarLevelOnsetPolyphonyMin" in names and "KeySignature" in names

    graph = _core.ConstraintGraph()
    graph.add_constraint(_core.GrammarConstraint())
    for i, t in enumerate(tokens):
        if i:
            assert not graph.get_mask(vocab)[t], "grammar masks %s after %s" % (names[i], " ".join(names[max(0, i - 6):i]))
        graph.step(t, vocab)
