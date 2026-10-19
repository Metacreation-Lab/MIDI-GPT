"""AR step prompts: a window after the first ends on the agent track's last
context bar, closed; and every AR track gets a step of its own."""

from __future__ import annotations

from pathlib import Path

import midigpt._core as _core
from midigpt._converters import to_cpp
from midigpt._types import Bar, Note, Score, Track
from midigpt.tokenizer import Tokenizer

CONFIG = Path(__file__).resolve().parents[3] / "models" / "yellow_config.json"


def _setup(model_dim):
    cfg = _core.EncoderConfig.from_json(CONFIG.read_text())
    cfg.model_dim = model_dim
    tok = Tokenizer(cfg)
    return cfg, tok, tok._vocab


def _score(n_tracks, n_bars):
    return Score(tracks=[
        Track(bars=[Bar(notes=[Note(pitch=60 + b % 12, velocity=100, onset_ticks=0, duration_ticks=12)])
                    for b in range(n_bars)])
        for _ in range(n_tracks)
    ])


def _mask(selected, ar):
    m = _core.SelectionMask()
    m.selected = selected
    m.autoregressive = ar
    m.ignore = [False] * len(selected)
    return m


def _names(vocab, tokens):
    return [str(vocab.decode(t)[0]).split(".")[-1] for t in tokens]


def test_later_ar_window_ends_on_a_closed_context_bar():
    cfg, tok, vocab = _setup(8)
    nb = 16
    steps = list(_core.StepPlanner(_mask([[True] * nb, [False] * nb], [True, False]), cfg, 4, 1).plan())
    step = steps[1]                                   # window [4, 12), generating bars 8-11
    assert (step.start_bar, step.end_bar) == (4, 12)
    state = _core.SessionState(
        to_cpp(tok.normalize_input(_score(2, nb))), step, vocab, _core.ConstraintGraph(),
        _core.Encoder(vocab), _core.Decoder(vocab), False, False, -1, -1, -1,
    )
    names = _names(vocab, state.context_tokens())
    agent = len(names) - 1 - names[::-1].index("Track")
    assert names[-1] == "BarEnd"
    assert names[agent:].count("Bar") == names[agent:].count("BarEnd") == 4


def test_each_ar_track_gets_its_own_step():
    cfg, _, _ = _setup(4)
    nb = 4
    steps = list(_core.StepPlanner(
        _mask([[True] * nb, [True] * nb, [False] * nb], [True, True, False]), cfg, 4, 3
    ).plan())
    assert [sorted({t for t, _ in s.bars_to_generate}) for s in steps] == [[0], [1]]
