"""The case list: the covering arrays (the first with the risky combinations) and the probes. Deterministic for a
number of seeds, so every shard, and status.py, computes the same one without running anything."""

from __future__ import annotations

from axes import AXES, DEFAULTS, MODELS, PROBES, risky, violates
from covering import covering_array


def all_cases(seeds: int) -> list[dict]:
    cases = []
    for s in range(seeds):
        for i, c in enumerate(covering_array(AXES, violates, risky() if s == 0 else (), seed=s)):
            cases.append({"id": f"s{s}-{i:04d}", "case": c, "expect": "ok"})
    for name, over in PROBES:
        for m in MODELS:
            if "model" in over and over["model"] != m:
                continue
            c = {**DEFAULTS, **over, "model": m}
            cases.append({"id": f"probe-{name}-{m}", "case": c, "expect": "rejected"})
    return cases


def shard_cases(cases: list[dict], model: str, shard: int, shards: int) -> list[dict]:
    mine = [c for c in cases if c["case"]["model"] == model]
    return [c for i, c in enumerate(mine) if i % shards == shard]
