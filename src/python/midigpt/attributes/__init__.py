from midigpt.attributes.base import AttributeAnalyzer, BaseAttribute
from midigpt.attributes.density import NoteDensity
from midigpt.attributes.polyphony import OnsetPolyphony
from midigpt.attributes.pitch_range import PitchRange
from midigpt.attributes.key_signature import KeySignature
from midigpt.attributes.note_duration import NoteDurationDistribution
from midigpt.attributes.silence import SilenceProportion
from midigpt.attributes.pitch_class_set import BarLevelPitchClassSet
from midigpt.attributes.quantile import (
    NoteDensityQuantile,
    PolyphonyQuantile,
    NoteDurationQuantile,
)

# Name → class registry used by AttributeAnalyzer.from_config().
# Encoder configs reference these names in the "attribute_controls" list.
ATTRIBUTE_REGISTRY = {
    "note_density":           NoteDensity,
    "onset_polyphony":        OnsetPolyphony,
    "pitch_range":            PitchRange,
    "key_signature":          KeySignature,
    "note_duration_dist":     NoteDurationDistribution,
    "silence_proportion":     SilenceProportion,
    "pitch_class_set":        BarLevelPitchClassSet,
    "note_density_quantile":  NoteDensityQuantile,
    "polyphony_quantile":     PolyphonyQuantile,
    "note_duration_quantile": NoteDurationQuantile,
}

__all__ = [
    "AttributeAnalyzer",
    "BaseAttribute",
    "ATTRIBUTE_REGISTRY",
    "NoteDensity",
    "OnsetPolyphony",
    "PitchRange",
    "KeySignature",
    "NoteDurationDistribution",
    "SilenceProportion",
    "BarLevelPitchClassSet",
    "NoteDensityQuantile",
    "PolyphonyQuantile",
    "NoteDurationQuantile",
]
