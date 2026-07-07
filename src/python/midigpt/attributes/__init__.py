from midigpt.attributes.base import AttributeAnalyzer, BaseAttribute
from midigpt.attributes.density import NoteDensity
from midigpt.attributes.key_signature import KeySignature
from midigpt.attributes.nomml import Nomml
from midigpt.attributes.note_duration import NoteDurationDistribution
from midigpt.attributes.pitch_class_set import BarLevelPitchClassSet, PitchClassSet
from midigpt.attributes.pitch_range import PitchRange
from midigpt.attributes.polyphony import OnsetPolyphony
from midigpt.attributes.quantile import (
    NoteDensityQuantile,
    NoteDurationQuantile,
    PolyphonyQuantile,
)
from midigpt.attributes.silence import SilenceProportion
from midigpt.attributes.tension import Tension, TensionDrum

# Name → class registry used by AttributeAnalyzer.from_config().
# Encoder configs reference these names in the "attribute_controls" list.
ATTRIBUTE_REGISTRY = {
    "note_density": NoteDensity,
    "onset_polyphony": OnsetPolyphony,
    "pitch_range": PitchRange,
    "key_signature": KeySignature,
    "note_duration_dist": NoteDurationDistribution,
    "silence_proportion": SilenceProportion,
    "pitch_class_set": PitchClassSet,
    "note_density_quantile": NoteDensityQuantile,
    "polyphony_quantile": PolyphonyQuantile,
    "note_duration_quantile": NoteDurationQuantile,
    "tension": Tension,
    "tension_drum": TensionDrum,
    "nomml": Nomml,
}

# TokenType (name) → (registry_key, params) used by AttributeAnalyzer
# auto-inference for bundles missing `attribute_controls_json`. The
# token-type names match the C++ TokenType enum; params are constructor
# kwargs for attributes that come in flavors (e.g. min/max, level).
TOKEN_TYPE_TO_ATTRIBUTE = {
    # track-level
    "MinPolyphony": ("polyphony_quantile", {"mode": "min"}),
    "MaxPolyphony": ("polyphony_quantile", {"mode": "max"}),
    "MinNoteDuration": ("note_duration_quantile", {"mode": "min"}),
    "MaxNoteDuration": ("note_duration_quantile", {"mode": "max"}),
    "NoteDensity": ("note_density_quantile", {}),
    # TrackLevelSilenceProportionMax/TrackLevelPitchRangeMax are the names
    # TokenType::name() actually returns at runtime -- SilenceProportion/
    # PitchRange are numeric aliases (types.h) of the same enum values but
    # are never what a switch-based name() resolves to, so keying on the
    # unprefixed names here silently never matched (both auto-inference in
    # base.py and BPE classification in tokenizer/bpe.py depend on this map).
    "TrackLevelSilenceProportionMax": ("silence_proportion", {}),
    "PitchClassSetTrack": ("pitch_class_set", {"level": "track"}),
    "TrackLevelPitchRangeMax": ("pitch_range", {}),
    "KeySignature": ("key_signature", {}),
    "NoteDurationDist": ("note_duration_dist", {}),
    "OnsetPolyphony": ("onset_polyphony", {}),
    # bar-level (C++ TokenType names)
    "BarLevelPitchClassSet": ("pitch_class_set", {}),
    "BarLevelOnsetDensity": ("note_density_quantile", {"level": "bar"}),
    "BarLevelOnsetPolyphonyMin": ("polyphony_quantile", {"mode": "min", "level": "bar"}),
    "BarLevelOnsetPolyphonyMax": ("polyphony_quantile", {"mode": "max", "level": "bar"}),
    "Tension": ("tension", {}),
    "TensionDrum": ("tension_drum", {}),
    "TrackLevelNomml": ("nomml", {}),
    # SilenceProportionBar, MinNoteDurationBar, MaxNoteDurationBar, PitchClassSetTrack:
    # no C++ token type yet — supported as Python classes for future models but not
    # auto-inferred from checkpoints until corresponding C++ entries are added.
}

__all__ = [
    "ATTRIBUTE_REGISTRY",
    "TOKEN_TYPE_TO_ATTRIBUTE",
    "AttributeAnalyzer",
    "BarLevelPitchClassSet",
    "BaseAttribute",
    "KeySignature",
    "NoteDensity",
    "NoteDensityQuantile",
    "NoteDurationDistribution",
    "NoteDurationQuantile",
    "Nomml",
    "OnsetPolyphony",
    "PitchClassSet",
    "PitchRange",
    "PolyphonyQuantile",
    "SilenceProportion",
    "Tension",
    "TensionDrum",
]
