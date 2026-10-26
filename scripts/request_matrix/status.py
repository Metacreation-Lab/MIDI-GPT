"""How far a matrix run is: python scripts/request_matrix/status.py <out dir> [--pull user@host:<remote out dir>]

With --pull, the shards' result files are copied from the cluster first (rsync, over an open SSH connection; they
are small). Then, per model: cases done of the total each shard was given, outcomes so far, and the time left at
the pace so far. The totals come from the case list itself (cases.py), so nothing runs on the cluster.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import all_cases, shard_cases

SHARDS = {"yellow_small": 2, "yellow_medium": 2, "prism_medium": 3, "expressive_medium": 6}   # as in the job spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pull", default=None, help="user@host:<remote out dir> to rsync the .jsonl files from")
    ap.add_argument("--ssh", default="ssh -o BatchMode=yes", help="ssh command for rsync (e.g. with -o ControlPath=...)")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--shards", default=None, help="model:n,... when the run used other shard counts")
    a = ap.parse_args()
    shards = dict(SHARDS)
    if a.shards:
        shards.update({k: int(v) for k, v in (p.split(":") for p in a.shards.split(","))})
    os.makedirs(a.out, exist_ok=True)
    if a.pull:
        subprocess.run(["rsync", "-az", "-e", a.ssh, "--include=*.jsonl", "--exclude=*",
                        a.pull.rstrip("/") + "/", a.out + "/"], check=True)

    cases = all_cases(a.seeds)
    print(f"{'shard':26s} {'done':>9s}  {'secs':>6s}  outcomes")
    totals = collections.Counter()
    left_s = {}
    for model, n in shards.items():
        for k in range(n):
            want = len(shard_cases(cases, model, k, n))
            path = os.path.join(a.out, f"{model}-{k}.jsonl")
            rows = [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []
            secs = sum(r.get("seconds") or 0 for r in rows)
            out = collections.Counter(r["outcome"] for r in rows)
            totals.update(out)
            pace = secs / len(rows) if rows else 0
            left_s[f"{model}-{k}"] = pace * (want - len(rows))
            mark = "done" if len(rows) >= want else ""
            print(f"{model + '-' + str(k):26s} {len(rows):4d}/{want:<4d}  {secs:6.0f}  "
                  f"{' '.join(f'{o}:{c}' for o, c in out.most_common())} {mark}")
    done = sum(totals.values())
    print(f"\n{done}/{len(cases)} cases: {' '.join(f'{o}:{c}' for o, c in totals.most_common())}")
    slowest = max(left_s.values()) if left_s else 0
    if slowest > 0:
        print(f"about {slowest / 60:.0f} min of generation left on the slowest shard (not counting retries or shrinking)")


if __name__ == "__main__":
    main()
