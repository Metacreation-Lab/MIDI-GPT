"""note_sink: what a streamed generation reports while it runs. Decodes are throttled (each one decodes the whole
step), so the step's end sends what is left: the notes streamed are exactly the notes generated."""

from __future__ import annotations

from conftest import drum_track, melodic_track

import midigpt.inference.session as session_mod
from midigpt._types import Score
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine


def _streamed(engine, score, tracks, monkeypatch, every_s):
    monkeypatch.setattr(session_mod, "STREAM_EVERY_S", every_s)
    cfg = InferenceConfig(model_dim=4, seed=0, max_attempts=1, novelty_check=False, silence_check=False,
                          density_hard_limit=4, mask_mode="attention")
    sess = engine.session(score, GenerationRequest(tracks=tracks, config=cfg))
    notes = []
    sess.note_sink = notes.extend
    return sess.run(), notes, sess


def _generated(result, sess):
    return sorted((t, b, n.pitch, n.onset_ticks) for t, bars in sess.generated_bars.items() for b in bars
                  for n in result.tracks[t].bars[b].notes)


def _key(notes):
    return sorted((n["track"], n["bar"], n["pitch"], n["onset_ticks"]) for n in notes)


def test_streamed_notes_are_the_generated_notes(tiny_gpt2, ghost_tokenizer, ghost_analyzer, monkeypatch):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = Score(tracks=[melodic_track(n_bars=4, notes_per_bar=2), melodic_track(n_bars=4, notes_per_bar=3)],
                  resolution=12)
    tracks = [TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[1, 2])]
    for every_s in (0.0, 3600.0):   # a decode after every note; only the one at the step's end
        result, notes, sess = _streamed(engine, score, tracks, monkeypatch, every_s)
        assert _key(notes) == _generated(result, sess), every_s


def test_drum_notes_are_streamed(tiny_gpt2, ghost_tokenizer, ghost_analyzer, monkeypatch):
    """A drum note has no NoteDuration token: a drum-only step must stream its notes all the same."""
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = Score(tracks=[melodic_track(n_bars=4), drum_track(n_bars=4)], resolution=12)
    result, notes, sess = _streamed(engine, score, [TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[1, 2])],
                                    monkeypatch, 0.0)
    assert _key(notes) == _generated(result, sess)


def test_a_streamed_generation_streams_with_batched_attempts(tiny_gpt2, ghost_tokenizer, ghost_analyzer, monkeypatch):
    """Batched attempts stream nothing, so a streamed generation runs its attempts one after the other."""
    monkeypatch.setattr(session_mod, "STREAM_EVERY_S", 0.0)
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = Score(tracks=[melodic_track(n_bars=4, notes_per_bar=2), melodic_track(n_bars=4, notes_per_bar=3)],
                  resolution=12)
    cfg = InferenceConfig(model_dim=4, seed=0, max_attempts=3, batch_attempts=True, novelty_check=False,
                          silence_check=False, density_hard_limit=4, mask_mode="attention")
    sess = engine.session(score, GenerationRequest(tracks=[TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=[1, 2])],
                                                   config=cfg))
    notes = []
    sess.note_sink = notes.extend
    result = sess.run()
    assert _key(notes) == _generated(result, sess) and notes
