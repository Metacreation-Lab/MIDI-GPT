"""Ignored tracks are left out of the prompt and come back unchanged in the result; run_variations
falls back to the sequential path (which steps model_dim down) when the batched prompt overflows."""

from __future__ import annotations

import copy

from conftest import melodic_track

import midigpt.inference.session as session_mod
from midigpt._types import Score
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine


class _Prompts:
    """TokenLogger stand-in that keeps each step's prompt."""

    def __init__(self):
        self.prompts = []

    def log_context(self, toks, vocab):
        self.prompts.append(list(toks))

    def __getattr__(self, name):
        return lambda *a, **k: None


def _request(tracks, **cfg):
    # density_hard_limit keeps the random model's bars short, so a later step's prompt cannot outgrow 512 positions
    base = dict(model_dim=4, seed=0, max_attempts=1, novelty_check=False, silence_check=False, density_hard_limit=4)
    base.update(cfg)
    return GenerationRequest(tracks=tracks, config=InferenceConfig(**base))


def _score():
    return Score(tracks=[melodic_track(n_bars=4, notes_per_bar=2), melodic_track(n_bars=4, notes_per_bar=3),
                         melodic_track(n_bars=4, notes_per_bar=2)], resolution=12)


def _run(engine, score, tracks):
    sess = engine.session(score, _request(tracks))
    sess.token_logger = _Prompts()
    notes = []
    sess.note_sink = notes.extend
    return sess.run(), sess.token_logger.prompts[0], notes


def test_ignored_track_is_not_in_the_prompt(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = _score()
    _, ignored, _ = _run(engine, score, [TrackPrompt(id=0, bars=[], ignore=True), TrackPrompt(id=1, bars=[]),
                                         TrackPrompt(id=2, bars=[1, 2])])
    _, dropped, _ = _run(engine, Score(tracks=score.tracks[1:], resolution=score.resolution),
                         [TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[1, 2])])
    assert ignored == dropped


def test_ignored_track_comes_back_and_ids_are_the_callers(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = _score()
    result, _, notes = _run(engine, score, [TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[], ignore=True),
                                            TrackPrompt(id=2, bars=[1, 2])])
    tok = engine._tokenizer
    expected = tok.normalize_output(tok.normalize_input(copy.deepcopy(score)))
    assert len(result.tracks) == 3
    assert result.tracks[1] == expected.tracks[1]
    assert result.tracks[0].bars == expected.tracks[0].bars
    assert {n["track"] for n in notes} <= {2}


def test_run_variations_falls_back_on_overflow(tiny_gpt2, ghost_tokenizer, ghost_analyzer, monkeypatch):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = _score()
    sess = engine.session(score, _request([TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[], ignore=True),
                                           TrackPrompt(id=2, bars=[1, 2])], num_candidates=2))

    def overflow(*a, **k):
        raise session_mod._ContextOverflow(600, 512)

    monkeypatch.setattr(sess, "_sample_step_batch", overflow)
    runs = []
    run = sess._run
    monkeypatch.setattr(sess, "_run", lambda: runs.append(1) or run())
    out = sess.run_variations()
    tok = engine._tokenizer
    expected = tok.normalize_output(tok.normalize_input(copy.deepcopy(score)))
    assert len(runs) == 2 and len(out) == 2
    assert out[0]["error"] is None
    assert all(len(c["score"].tracks) == 3 and c["score"].tracks[1] == expected.tracks[1]
               for c in out if c["score"] is not None)
