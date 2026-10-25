"""Summarises a matrix run: python scripts/request_matrix/report.py <out dir> [--md report.md]

Outcomes per model, then every failure grouped by signature (the error and where it was raised, or the checks a
reply broke), with how often, on which models, the smallest case that still fails (shrunk) and one failure file
to replay. Warnings are counted by check.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

BAD = ("crash", "violation", "timeout", "no_take", "checks_failed")


def load(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def is_failure(r):
    if r["expect"] == "rejected":
        return r["outcome"] not in ("rejected", "skipped")
    return r["outcome"] in BAD or r["outcome"] == "rejected"


def render(rows) -> str:
    models = sorted({r["model"] for r in rows})
    outcomes = ["ok", "violation", "crash", "checks_failed", "timeout", "no_take", "rejected", "skipped"]
    lines = ["# Request matrix", "", f"{len(rows)} cases.", "",
             "| model | " + " | ".join(outcomes) + " | median s |", "|---" * (len(outcomes) + 2) + "|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        c = Counter(r["outcome"] for r in rs)
        secs = sorted(r["seconds"] for r in rs if r.get("seconds"))
        med = f"{secs[len(secs) // 2]:.1f}" if secs else "-"
        lines.append(f"| {m} | " + " | ".join(str(c.get(o, 0)) for o in outcomes) + f" | {med} |")

    groups = defaultdict(list)
    for r in rows:
        if is_failure(r):
            groups[r.get("signature") or r["outcome"]].append(r)
    lines += ["", f"## Failures: {len(groups)} signatures", ""]
    for sig, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        shrunk = next((r["shrunk"] for r in rs if r.get("shrunk")), None)
        lines += [f"### {sig}", "",
                  f"- {len(rs)} cases, outcome {sorted({r['outcome'] for r in rs})}, models {sorted({r['model'] for r in rs})}",
                  f"- smallest: `{json.dumps(shrunk)}`" if shrunk else "- (not shrunk)",
                  f"- replay: `failures/{rs[0]['id']}-{rs[0]['model']}.json`"]
        if rs[0].get("message"):
            lines.append(f"- message: {rs[0]['message'][:300]}")
        detail = [f for f in rs[0].get("findings", []) if f[0] == "violation"][:3]
        lines += [f"- {f[1]}: {f[2]}" for f in detail]
        # the axis values every failing case shares: a hint at the cause
        common = {k: v for k, v in rs[0]["case"].items() if all(r["case"].get(k) == v for r in rs)} if len(rs) > 1 else {}
        if common:
            lines.append(f"- every case has: `{json.dumps(common)}`")
        lines.append("")

    warn = Counter((f[1], r["model"]) for r in rows for f in r.get("findings", []) if f[0] == "warning")
    if warn:
        lines += ["## Warnings", "", "| check | model | cases |", "|---|---|---|"]
        lines += [f"| {k} | {m} | {n} |" for (k, m), n in sorted(warn.items())]
    rejected = Counter(r.get("signature") for r in rows if r["expect"] == "ok" and r["outcome"] == "rejected")
    if rejected:
        lines += ["", "## Valid-looking cases the server refused", ""] + [f"- {n} x {s}" for s, n in rejected.most_common()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    text = render(load(a.out))
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
