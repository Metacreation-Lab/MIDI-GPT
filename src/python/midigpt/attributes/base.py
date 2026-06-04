from abc import ABC, abstractmethod

from midigpt._types import Score


class BaseAttribute(ABC):
    name: str  # e.g. "note_density"
    token_type: str  # e.g. "NoteDensity"
    level: str  # "track" | "bar"
    track_type: str  # "melodic" | "drum" | "both"
    size: int  # token-domain size for this attribute's vocab slot

    @abstractmethod
    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int: ...

    @abstractmethod
    def quantize(self, value: float | int) -> int: ...

    def dequantize(self, quantized: int) -> float | int:
        raise NotImplementedError

    def achievable_range(
        self, fixed_score: "Score", track_idx: int, generated_bars: list[int]
    ) -> tuple[int, int]:
        """Inclusive closed interval [min_q, max_q] of quantized values that
        a track-level override could still realize given the fixed (non-
        generated) bars in `fixed_score` and the indices of the bars that
        will be (re)generated.

        Default = full domain (no achievability constraint). Per-attribute
        subclasses override this for monotonic attributes (min_*/max_*) and
        averaged attributes (density). Used by validation to emit a warning
        when an override is physically infeasible — never to reject.
        """
        size = int(getattr(self, "size", 0))
        return (0, max(0, size - 1))


class AttributeAnalyzer:
    def __init__(self, attributes: list[BaseAttribute]):
        self._attrs = {a.name: a for a in attributes}

    def compute_track_tokens(self, score: Score, track_idx: int) -> dict[str, int]:
        result = {}
        track = score.tracks[track_idx]
        from midigpt._core import TrackType

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
            {name: a.compute(score, i) for name, a in self._attrs.items() if a.level == "track"}
            for i in range(len(score.tracks))
        ]

    def evaluate(
        self, requested: dict[str, int], realized_score: Score, track_idx: int
    ) -> dict[str, float]:
        result = {}
        for name, req_q in requested.items():
            attr = self._attrs.get(name)
            if attr is None:
                continue
            raw = attr.compute(realized_score, track_idx)
            real_q = attr.quantize(raw)
            result[name] = 1.0 if real_q == req_q else 0.0
        return result

    def attribute_sizes(self) -> dict[str, int]:
        """Return {attribute_name: size, ...} — value-domain size for each
        configured attribute. Validators use this to range-check user input."""
        return {a.name: int(getattr(a, "size", 0)) for a in self._attrs.values()}

    def attribute_levels(self) -> dict[str, str]:
        """Return {attribute_name: "track"|"bar"} — used by validators to
        enforce that bar-level attributes only appear in `bar_attributes`
        and track-level only in `attributes`."""
        return {a.name: getattr(a, "level", "track") for a in self._attrs.values()}

    def get(self, name: str):
        """Lookup an attribute instance by name, or None."""
        return self._attrs.get(name)

    def attribute_track_types(self) -> dict[str, str]:
        """Return {attribute_name: "melodic"|"drum"|"both"} so callers can
        decide whether a control applies to a given track."""
        return {a.name: getattr(a, "track_type", "both") for a in self._attrs.values()}

    def attribute_value_labels(self) -> dict[str, list[str]]:
        """Return {attribute_name: [label_for_bin_0, ...]} for every attribute
        that defines `value_labels()`. Attributes without labels are omitted —
        UIs can fall back to numeric values."""
        out = {}
        for a in self._attrs.values():
            fn = getattr(a, "value_labels", None)
            if callable(fn):
                try:
                    labels = list(fn())
                    if labels:
                        out[a.name] = labels
                except Exception:
                    pass
        return out

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
                raise ValueError(f"Attribute '{attr.name}' has no .size; cannot build vocab")
            specs.append((attr.token_type, int(size)))
        return specs

    def report(
        self,
        result_score: Score,
        track_idx: int,
        requested: dict[str, int] | None = None,
    ) -> dict:
        """Compare sampled attribute tokens against realized (note-derived) values.

        Returns a dict with two keys:
          "track": {attr_name: {"sampled", "realized", "requested", "consistent", "achieved"}}
          "bars":  [{attr_name: {...}}, ...]  — one dict per bar

        "sampled"    = quantized value the model emitted (from track.attributes).
        "realized"   = quantized value computed from the actual notes.
        "requested"  = quantized value the caller forced / requested (if provided).
        "consistent" = sampled == realized.
        "achieved"   = realized == requested (None when no request was made).
        """
        if track_idx >= len(result_score.tracks):
            return {"track": {}, "bars": []}

        track = result_score.tracks[track_idx]
        attrs_map = track.attributes  # dict[str, int] populated by decoder

        def _entry(sampled, realized, req):
            return {
                "sampled": sampled,
                "realized": realized,
                "requested": req,
                "consistent": sampled == realized if sampled is not None else None,
                "achieved": (realized == req) if req is not None else None,
            }

        # --- track-level ---
        track_report: dict[str, dict] = {}
        for attr in self._attrs.values():
            if attr.level != "track":
                continue
            sampled = attrs_map.get(attr.name)
            raw = attr.compute(result_score, track_idx)
            realized = attr.quantize(raw)
            req = (requested or {}).get(attr.name)
            track_report[attr.name] = _entry(sampled, realized, req)

        # --- bar-level ---
        n_bars = len(track.bars)
        bar_reports: list[dict[str, dict]] = [{} for _ in range(n_bars)]
        bar_attrs = [a for a in self._attrs.values() if a.level == "bar"]
        req_bars = (requested or {}).get("bar_attributes") or {}
        for bar_idx in range(n_bars):
            for attr in bar_attrs:
                key = f"bar_{attr.token_type}_{bar_idx}"
                sampled = attrs_map.get(key)
                raw = attr.compute(result_score, track_idx, bar_idx)
                realized = attr.quantize(raw)
                req_bar = req_bars.get(bar_idx, {}).get(attr.name)
                bar_reports[bar_idx][attr.name] = _entry(sampled, realized, req_bar)

        return {"track": track_report, "bars": bar_reports}

    @staticmethod
    def from_config(config) -> "AttributeAnalyzer":
        """Build an analyzer from an EncoderConfig.

        `config.token_domains` is the source of truth: every TokenType that
        maps to a registry key in `TOKEN_TYPE_TO_ATTRIBUTE` becomes an
        attribute with default params.

        `config.attribute_controls_json`, when present, is an optional
        overlay. Each entry is `{"name": "<registry_key>", "params": {...}}`.
        If the registry key matches one already auto-inferred, the explicit
        entry replaces it (override with custom params); otherwise the entry
        is appended (addition).
        """
        import json

        from midigpt.attributes import ATTRIBUTE_REGISTRY, TOKEN_TYPE_TO_ATTRIBUTE

        try:
            cfg_dict = json.loads(config.to_json())
        except Exception:
            cfg_dict = {}

        attrs: list = []
        # Keyed by token_type string — the unique identity of each attribute slot
        # in the model vocabulary. Both the auto-inference and explicit-override
        # paths use this so that different registry entries (e.g. min/max polyphony)
        # that share a registry key but have distinct token_types don't collide.
        by_token_type: dict[str, int] = {}

        for d in cfg_dict.get("token_domains", []) or []:
            tt = d.get("type")
            mapping = TOKEN_TYPE_TO_ATTRIBUTE.get(tt)
            if not mapping:
                continue
            reg_key, params = mapping
            if reg_key not in ATTRIBUTE_REGISTRY:
                continue
            by_token_type[tt] = len(attrs)
            attrs.append(ATTRIBUTE_REGISTRY[reg_key](**params))

        raw = getattr(config, "attribute_controls_json", "") or ""
        if raw:
            try:
                entries = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as e:
                raise ValueError(f"attribute_controls_json is not valid JSON: {e}") from e
            for entry in entries or []:
                name = entry["name"]
                params = entry.get("params", {})
                if name not in ATTRIBUTE_REGISTRY:
                    raise KeyError(
                        f"Unknown attribute control '{name}'. "
                        f"Registered: {sorted(ATTRIBUTE_REGISTRY.keys())}"
                    )
                inst = ATTRIBUTE_REGISTRY[name](**params)
                if inst.token_type in by_token_type:
                    attrs[by_token_type[inst.token_type]] = inst
                else:
                    by_token_type[inst.token_type] = len(attrs)
                    attrs.append(inst)

        return AttributeAnalyzer(attrs)
