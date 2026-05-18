from abc import ABC, abstractmethod
from typing import Optional
from midigpt_refactor._types import Score

class BaseAttribute(ABC):
    name:       str   # e.g. "note_density"
    token_type: str   # e.g. "NoteDensity"
    level:      str   # "track" | "bar"
    track_type: str   # "melodic" | "drum" | "both"
    size:       int   # token-domain size for this attribute's vocab slot

    @abstractmethod
    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int: ...

    @abstractmethod
    def quantize(self, value: float | int) -> int: ...

    def dequantize(self, quantized: int) -> float | int:
        raise NotImplementedError

class AttributeAnalyzer:
    def __init__(self, attributes: list[BaseAttribute]):
        self._attrs = {a.name: a for a in attributes}

    def compute_track_tokens(self, score: Score, track_idx: int) -> dict[str, int]:
        result = {}
        track  = score.tracks[track_idx]
        from midigpt_refactor._core import TrackType
        if hasattr(track, "track_type"):
            is_drum = track.track_type == "drum"
        else:
            is_drum = track.type == TrackType.Drum
        for attr in self._attrs.values():
            if attr.level != "track":
                continue
            if attr.track_type == "melodic" and is_drum:
                continue
            if attr.track_type == "drum" and not is_drum:
                continue
            raw = attr.compute(score, track_idx)
            result[attr.name] = attr.quantize(raw)
        return result

    def compute_bar_tokens(self, score: Score, track_idx: int, bar_idx: int) -> dict[str, int]:
        return {
            attr.token_type: attr.quantize(attr.compute(score, track_idx, bar_idx))
            for attr in self._attrs.values()
            if attr.level == "bar"
        }

    def compute_all(self, score: Score) -> list[dict[str, float | int]]:
        return [
            {name: a.compute(score, i)
             for name, a in self._attrs.items()
             if a.level == "track"}
            for i in range(len(score.tracks))
        ]

    def evaluate(self, requested: dict[str, int], realized_score: Score, track_idx: int) -> dict[str, float]:
        result = {}
        for name, req_q in requested.items():
            attr = self._attrs.get(name)
            if attr is None:
                continue
            raw      = attr.compute(realized_score, track_idx)
            real_q   = attr.quantize(raw)
            result[name] = 1.0 if real_q == req_q else 0.0
        return result

    def attribute_sizes(self) -> dict[str, int]:
        """Return {attribute_name: size, ...} — value-domain size for each
        configured attribute. Validators use this to range-check user input."""
        return {a.name: int(getattr(a, "size", 0)) for a in self._attrs.values()}

    def token_domain_specs(self) -> list[tuple[str, int]]:
        """Return [(token_type_name, size), ...] for every attribute.

        Used to extend the C++ EncoderConfig.token_domains before the
        Vocabulary is built. Python is the source of truth for both
        token_type and size on attribute controls.
        """
        specs = []
        for attr in self._attrs.values():
            size = getattr(attr, "size", None)
            if size is None:
                raise ValueError(
                    f"Attribute '{attr.name}' has no .size; cannot build vocab")
            specs.append((attr.token_type, int(size)))
        return specs

    @staticmethod
    def from_config(config) -> "AttributeAnalyzer":
        """Build an analyzer from an EncoderConfig.

        Reads `config.attribute_controls_json` (a JSON list) and resolves
        each entry through `ATTRIBUTE_REGISTRY`. An empty list means no
        controls — the analyzer is valid but emits nothing.

        Each entry is a dict: `{"name": "<registry_key>", "params": {...}}`.
        `params` is optional and is forwarded to the class constructor.
        """
        import json
        from midigpt_refactor.attributes import ATTRIBUTE_REGISTRY
        raw = getattr(config, "attribute_controls_json", "[]")
        try:
            entries = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            raise ValueError(f"attribute_controls_json is not valid JSON: {e}")
        attrs = []
        for entry in entries:
            name = entry["name"]
            params = entry.get("params", {})
            if name not in ATTRIBUTE_REGISTRY:
                raise KeyError(
                    f"Unknown attribute control '{name}'. "
                    f"Registered: {sorted(ATTRIBUTE_REGISTRY.keys())}"
                )
            attrs.append(ATTRIBUTE_REGISTRY[name](**params))
        return AttributeAnalyzer(attrs)
