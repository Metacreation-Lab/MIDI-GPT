import midigpt._core as _core
from midigpt._types import Score, Note
from midigpt._converters import to_cpp, from_cpp
from midigpt.attributes.base import AttributeAnalyzer


def resample_delta(score: Score, source_res: int, target_res: int) -> Score:
    """Resample note timings from source to target resolution, applying deltas.

    For each note, new_time = (target_res * onset_ticks / source_res) + delta,
    clamped to >= 0. Duration is scaled proportionally.
    """
    if source_res == target_res and all(
        n.delta == 0 for t in score.tracks for b in t.bars for n in b.notes
    ):
        return score

    score.resolution = target_res
    for track in score.tracks:
        for bar in track.bars:
            for note in bar.notes:
                new_onset = int(target_res * note.onset_ticks / source_res)
                new_onset = max(0, new_onset + note.delta)
                new_dur = max(1, int(target_res * note.duration_ticks / source_res))
                note.onset_ticks = new_onset
                note.duration_ticks = new_dur
                note.delta = 0
    return score

class Tokenizer:
    def __init__(self, encoder_config: _core.EncoderConfig,
                 analyzer: AttributeAnalyzer | None = None):
        analyzer = analyzer or AttributeAnalyzer.from_config(encoder_config)
        # Inject attribute-control token domains into the config before
        # building the vocab. Python is the source of truth for these sizes.
        specs = analyzer.token_domain_specs()
        if specs:
            encoder_config.add_attribute_token_domains(specs)
        self._vocab    = _core.Vocabulary(encoder_config)
        self._encoder  = _core.Encoder(self._vocab)
        self._decoder  = _core.Decoder(self._vocab)
        self._analyzer = analyzer

    def encode(
        self,
        score: Score,
        opts: "_core.EncodeOptions | None" = None,
        compute_attributes: bool = True,
    ) -> list[int]:
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
        cfg = self._vocab.config()
        if resample and cfg.resolution != cfg.decode_resolution:
            score = resample_delta(score, cfg.resolution, cfg.decode_resolution)
        return score

    def vocab_size(self) -> int:
        return self._vocab.size()

    @classmethod
    def from_checkpoint_bundle(cls, encoder_config, analyzer) -> "Tokenizer":
        return cls(encoder_config, analyzer)

    @classmethod
    def from_checkpoint(cls, path: str,
                        analyzer: AttributeAnalyzer | None = None) -> "Tokenizer":
        from midigpt.tokenizer.checkpoint import load_checkpoint
        bundle = load_checkpoint(path)
        return cls(bundle.encoder_config, analyzer)
