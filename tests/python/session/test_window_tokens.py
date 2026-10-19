"""window_tokens() is the current step's prompt plus what that step generated, so it stays within the
model's context across steps; context_len + gen_count (gen_count runs over the whole session) does not."""

from __future__ import annotations

import torch
from conftest import melodic_track

from midigpt._types import Score
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine


def test_window_tokens_stay_within_the_context_across_steps(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    torch.manual_seed(0)
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = Score(tracks=[melodic_track(n_bars=8, notes_per_bar=1), melodic_track(n_bars=8, notes_per_bar=0)],
                  resolution=12)
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[]), TrackPrompt(id=1, bars=list(range(8)), autoregressive=True)],
        config=InferenceConfig(model_dim=4, bars_per_step=4, seed=0, max_attempts=1,
                               novelty_check=False, silence_check=False))
    sess = engine.session(score, req)
    seen = []                                   # (prompt, session gen_count, window) after each step's decode
    sample_step = sess._sample_step

    def spy(*a, **k):
        try:
            return sample_step(*a, **k)
        finally:
            seen.append((sess.context_len, sess.gen_count, sess.window_tokens()))

    sess._sample_step = spy
    sess.run()

    prompts = {ctx for ctx, _, _ in seen}
    assert len(prompts) >= 2, "expected progress from more than one step"
    assert all(w <= sess.max_context_len for _, _, w in seen)
    assert sess.peak_window_tokens() == max([w for _, _, w in seen] + [sess.window_tokens()])
    assert sess.peak_window_tokens() <= sess.max_context_len
    # the old readout: the second step's prompt plus every token so far
    assert max(ctx + gen for ctx, gen, _ in seen) > sess.peak_window_tokens()
