"""A pairwise covering array over AXES, with required combinations: every pair of values of two axes appears in
some case (unless no valid case can hold it), and every combination in `required` appears whole. Greedy, in the
AETG style: each new case starts from something still uncovered and fills the other axes with the value that
covers the most uncovered pairs. Deterministic for a given seed, so every shard computes the same list.
"""

from __future__ import annotations

import itertools
import random


def _pairs(case: dict, names: list[str]) -> set:
    keys = [n for n in names if n in case]
    return {(a, case[a], b, case[b]) for a, b in itertools.combinations(keys, 2)}


def covering_array(axes: dict, violates, required=(), seed: int = 0, tries: int = 30) -> list[dict]:
    rng = random.Random(seed)
    names = list(axes)
    uncovered = set()
    for a, b in itertools.combinations(names, 2):
        for va in axes[a]:
            for vb in axes[b]:
                if violates({a: va, b: vb}) is None:
                    uncovered.add((a, va, b, vb))
    pending = [dict(r) for r in required if violates(r) is None]
    cases = []

    def complete(seedcase):
        case = dict(seedcase)
        order = [n for n in names if n not in case]
        rng.shuffle(order)
        for n in order:
            best, best_gain = [], -1
            for v in axes[n]:
                trial = dict(case, **{n: v})
                if violates(trial) is not None:
                    continue
                gain = sum(1 for m in case if ((m, case[m], n, v) if names.index(m) < names.index(n)
                                                  else (n, v, m, case[m])) in uncovered)
                if gain > best_gain:
                    best, best_gain = [v], gain
                elif gain == best_gain:
                    best.append(v)
            if not best:
                return None
            case[n] = rng.choice(best)
        return case

    while pending or uncovered:
        if pending:
            start = [pending.pop(0)]
        else:
            pool = sorted(uncovered, key=repr)
            start = []
            for a, va, b, vb in rng.sample(pool, min(tries, len(pool))):
                start.append({a: va, b: vb})
        best, best_gain = None, -1
        for s in start:
            for _ in range(max(1, tries // len(start))):
                c = complete(s)
                if c is None:
                    continue
                gain = len(_pairs(c, names) & uncovered)
                if gain > best_gain:
                    best, best_gain = c, gain
        if best is None:
            # nothing valid holds what was tried: give those pairs up rather than loop
            for s in start:
                if len(s) == 2:
                    (a, va), (b, vb) = s.items()
                    uncovered.discard((a, va, b, vb))
            continue
        uncovered -= _pairs(best, names)
        cases.append(best)
    return cases
