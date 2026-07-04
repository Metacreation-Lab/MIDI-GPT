import copy

import midigpt._core as _core
from midigpt._converters import from_cpp, to_cpp
from midigpt._types import Score
from midigpt.attributes.base import AttributeAnalyzer


def resample_delta(score: Score, source_res: int, target_res: int, use_delta: bool) -> Score:
    """Resample note timings from source to target resolution.

    ``use_delta`` gates whether ``note.delta`` (a microtiming residual, in
    units of 1/source_res of one source_res grid cell) is folded into the
    rescale. When False, delta is ignored entirely and onset is truncated the
    same way regardless of whatever delta an upstream reader populated -- this
    is required so configs with emit_delta_tokens=false get byte-identical
    output to before microtiming residuals existed.

    When True, the true continuous position (onset_ticks + delta/source_res)
    is rescaled to target_res and truncated; the new leftover fraction becomes
    the note's delta, again in units of 1/target_res of one target_res cell.
    """
    if source_res == target_res and all(
        n.delta == 0 for t in score.tracks for b in t.bars for n in b.notes
    ):
        return score

    scale = target_res / source_res
    score.resolution = target_res
    for track in score.tracks:
        for bar in track.bars:
            for note in bar.notes:
                if use_delta:
                    true_pos = (note.onset_ticks + note.delta / source_res) * scale
                    new_onset = int(true_pos)
                    new_onset = max(0, new_onset)
                    residual = true_pos - new_onset
                    note.onset_ticks = new_onset
                    note.delta = int(round(residual * target_res))
                else:
                    new_onset = max(0, int(target_res * note.onset_ticks / source_res))
                    note.onset_ticks = new_onset
                    note.delta = 0
                note.duration_ticks = max(1, int(target_res * note.duration_ticks / source_res))
    return score


class Tokenizer:
    def __init__(
        self, encoder_config: _core.EncoderConfig, analyzer: AttributeAnalyzer | None = None
    ):
        analyzer = analyzer or AttributeAnalyzer.from_config(encoder_config)
        # Inject attribute-control token domains into the config before
        # building the vocab. Python is the source of truth for these sizes.
        specs = analyzer.token_domain_specs()
        if specs:
            encoder_config.add_attribute_token_domains(specs)
        self._vocab = _core.Vocabulary(encoder_config)
        self._encoder = _core.Encoder(self._vocab)
        self._decoder = _core.Decoder(self._vocab)
        self._analyzer = analyzer

    def normalize_input(self, score: Score) -> Score:
        """Return score resampled to model resolution.

        If the score is already at model resolution, the original object is
        returned unchanged.  Otherwise a deep copy is made and resampled so
        the caller's score is never mutated.
        """
        cfg = self._vocab.config()
        if score.resolution == cfg.resolution:
            return score
        return resample_delta(
            copy.deepcopy(score), score.resolution, cfg.resolution,
            use_delta=cfg.emit_delta_tokens,
        )

    def normalize_output(self, score: Score) -> Score:
        """Return score resampled from model resolution to decode_resolution.

        The C++ decoder always outputs at cfg.resolution (model-internal PPQ).
        This resamples to cfg.decode_resolution so callers receive notes at the
        documented output resolution.  If they are equal, the original object
        is returned unchanged.
        """
        cfg = self._vocab.config()
        if cfg.resolution == cfg.decode_resolution:
            return score
        return resample_delta(
            score, cfg.resolution, cfg.decode_resolution,
            use_delta=cfg.emit_delta_tokens,
        )

    def encode(
        self,
        score: Score,
        opts: "_core.EncodeOptions | None" = None,
        compute_attributes: bool = True,
    ) -> list[int]:
        score = self.normalize_input(score)
        if compute_attributes and self._analyzer:
            for t_idx, track in enumerate(score.tracks):
                # Pybind11 std::map returns a copy, so we must assign the whole dict back
                new_attrs = dict(track.attributes)
                attrs = self._analyzer.compute_track_tokens(score, t_idx)
                for key, val in attrs.items():
                    new_attrs[key] = val
                for b_idx in range(len(track.bars)):
                    bar_attrs = self._analyzer.compute_bar_tokens(score, t_idx, b_idx)
                    for key, val in bar_attrs.items():
                        new_attrs[f"bar_{key}_{b_idx}"] = val
                track.attributes = new_attrs
        return self._encoder.encode(to_cpp(score), opts or _core.EncodeOptions())

    def decode(self, tokens: list[int], resample: bool = True) -> Score:
        score = from_cpp(self._decoder.decode(tokens))
        if resample:
            score = self.normalize_output(score)
        return score

    def vocab_size(self) -> int:
        return self._vocab.size()

    @classmethod
    def from_checkpoint_bundle(cls, encoder_config, analyzer) -> "Tokenizer":
        return cls(encoder_config, analyzer)

    @classmethod
    def from_checkpoint(cls, path: str, analyzer: AttributeAnalyzer | None = None) -> "Tokenizer":
        from midigpt.tokenizer.checkpoint import load_checkpoint

        bundle = load_checkpoint(path)
        return cls(bundle.encoder_config, analyzer)
