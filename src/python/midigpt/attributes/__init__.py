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
from midigpt.attributes.tension import Tension, TensionDrum

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
    "tension":                Tension,
    "tension_drum":           TensionDrum,
}

# TokenType (name) → (registry_key, params) used by AttributeAnalyzer
# auto-inference for bundles missing `attribute_controls_json`. The
# token-type names match the C++ TokenType enum; params are constructor
# kwargs for attributes that come in flavors (e.g. min/max).
TOKEN_TYPE_TO_ATTRIBUTE = {
    "MinPolyphony":    ("polyphony_quantile",     {"mode": "min"}),
    "MaxPolyphony":    ("polyphony_quantile",     {"mode": "max"}),
    "MinNoteDuration": ("note_duration_quantile", {"mode": "min"}),
    "MaxNoteDuration": ("note_duration_quantile", {"mode": "max"}),
    "NoteDensity":     ("note_density_quantile",  {}),
    "Tension":         ("tension",                {}),
    "TensionDrum":     ("tension_drum",           {}),
    "PitchClassSet":   ("pitch_class_set",        {}),
    "PitchRange":      ("pitch_range",            {}),
    "KeySignature":    ("key_signature",          {}),
    "NoteDurationDist":("note_duration_dist",     {}),
    "SilenceProportion":("silence_proportion",    {}),
    "OnsetPolyphony":  ("onset_polyphony",        {}),
}

__all__ = [
    "AttributeAnalyzer",
    "BaseAttribute",
    "ATTRIBUTE_REGISTRY",
    "TOKEN_TYPE_TO_ATTRIBUTE",
    "Tension",
    "TensionDrum",
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
