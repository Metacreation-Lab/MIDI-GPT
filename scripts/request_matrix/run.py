"""Runs the request matrix against one model: every case of the covering array (axes.py, covering.py) for that
model, plus the probes, through InferenceEngine.session() the way the HTTP server calls it. Writes one JSON line
per case to <out>/<model>-<shard>.jsonl, and each failing request (score + request) to <out>/failures/ so it can
be replayed. Failures are shrunk: axes are put back to their defaults one at a time while the failure persists.

    python scripts/request_matrix/run.py --data <GigaMIDI test dir or .mid folder> --model prism_medium \
        --out runs/matrix [--shard 0 --shards 4] [--limit 5] [--device cuda]

The case list depends only on --seeds, so every shard computes the same one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from axes import AXES, DEFAULTS, MODELS, PROBES, risky, violates
from build import build, load_pieces
from checks import check, same_result
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


def signature(exc: BaseException) -> str:
    frames = [fr for fr in traceback.extract_tb(exc.__traceback__) if "midigpt" in fr.filename]
    where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].name}" if frames else "?"
    msg = re.sub(r"\d+(\.\d+)?", "N", str(exc).splitlines()[0] if str(exc) else "")[:160]
    return f"{type(exc).__name__} @ {where}: {msg}"


class Runner:
    def __init__(self, engine, model, pieces, timeout):
        self.engine, self.model, self.pieces, self.timeout = engine, model, pieces, timeout

    def once(self, case, seed, keep=False):
        """One generation: a result dict (outcome, signature, findings, ...)."""
        from midigpt._types import Score
        from midigpt.inference import GenerationRequest
        from midigpt.inference.validation import RequestValidationError

        r = {"outcome": None}
        try:
            score, req, info = build(case, self.engine, self.pieces, seed)
        except LookupError as exc:
            return {"outcome": "skipped", "message": str(exc)}
        r["info"] = info
        if keep:
            r["score"], r["request"] = score.to_dict(), req
        t0 = time.time()
        streamed = [] if case.get("stream") else None
        try:
            # through dicts, like the server: catches whatever only breaks on parsing
            sess = self.engine.session(Score.from_dict(score.to_dict()), GenerationRequest.from_dict(json.loads(json.dumps(req))))
        except RequestValidationError as exc:
            r.update(outcome="rejected", message=str(exc)[:400], signature=signature(exc))
            return r
        except Exception as exc:
            r.update(outcome="crash", stage="validate", signature=signature(exc), message=str(exc)[:400],
                     traceback=traceback.format_exc()[-3000:])
            return r
        if streamed is not None:
            sess.note_sink = lambda notes: streamed.extend(notes)
            sess.progress_sink = lambda: None
        sess.cancel_event = threading.Event()
        timer = threading.Timer(self.timeout, sess.cancel_event.set)
        timer.start()
        try:
            if case["takes"] > 1:
                variations = sess.run_variations()
                outs = [v["score"] for v in variations if v["score"] is not None]
                r["takes"] = {"asked": case["takes"], "got": len(variations), "ok": len(outs),
                              "errors": sorted({str(v["error"])[:120] for v in variations if v["error"]})}
                out = outs[0] if outs else None
            else:
                out = sess.run()
        except Exception as exc:
            timer.cancel()
            timed_out = sess.cancel_event.is_set()
            checks = isinstance(exc, RuntimeError) and str(exc).startswith("Max attempts")
            r.update(outcome="timeout" if timed_out else "checks_failed" if checks else "crash", stage="generate", signature=signature(exc),
                     message=str(exc)[:400], traceback=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 2))
            return r
        timer.cancel()
        r.update(seconds=round(time.time() - t0, 2), gen_count=sess.gen_count, attempts=sess.attempt,
                 truncated=bool(getattr(sess, "truncated", False)), dropped_context=list(sess.dropped_context or []),
                 generated_bars={str(k): v for k, v in (sess.generated_bars or {}).items()},
                 peak_context=getattr(sess, "peak_context_len", None), streamed=None if streamed is None else len(streamed))
        if out is None:
            r.update(outcome="no_take", signature="every take failed: " + "; ".join(r["takes"]["errors"])[:200])
            return r
        findings = check(case, req, score, out, sess, streamed)
        r["findings"] = findings
        hard = [x for x in findings if x[0] == "violation"]
        r["outcome"] = "violation" if hard else "ok"
        if hard:
            r["signature"] = "violation: " + ", ".join(sorted({x[1] for x in hard}))
        r["_out"] = out
        return r

    def shrink(self, case, seed, sig, budget=12):
        """The case with as many axes back at their defaults as keep the same failure."""
        cur = dict(case)
        for k in AXES:
            if budget <= 0:
                break
            if k == "model" or cur[k] == DEFAULTS[k]:
                continue
            trial = dict(cur, **{k: DEFAULTS[k]})
            if violates(trial) is not None:
                continue
            budget -= 1
            r = self.once(trial, seed)
            if r.get("signature") == sig:
                cur = trial
        return {k: v for k, v in cur.items() if v != DEFAULTS[k] or k == "model"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=2, help="covering arrays to run (the first one has the risky combinations)")
    ap.add_argument("--limit", type=int, default=0, help="run only this many cases (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds before a generation is cancelled")
    a = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    import logging
    logging.getLogger("midigpt").setLevel(logging.ERROR)   # the achievable-range notes would fill the log
    from midigpt.inference import InferenceEngine

    os.makedirs(os.path.join(a.out, "failures"), exist_ok=True)
    cases = [c for c in all_cases(a.seeds) if c["case"]["model"] == a.model]
    cases = [c for i, c in enumerate(cases) if i % a.shards == a.shard]
    if a.limit:
        cases = cases[: a.limit]
    print(f"{a.model}: {len(cases)} cases (shard {a.shard}/{a.shards})", flush=True)

    t0 = time.time()
    pieces = load_pieces(a.data)
    print("pieces:", {k: [p for p, _ in v] for k, v in pieces.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    engine = InferenceEngine.from_pretrained(a.model, device=a.device)
    runner = Runner(engine, a.model, pieces, a.timeout)

    path = os.path.join(a.out, f"{a.model}-{a.shard}.jsonl")
    done = set()
    if os.path.exists(path):   # resume a requeued job
        with open(path) as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}
    shrunk: dict[str, int] = {}
    with open(path, "a") as fh:
        for n, c in enumerate(cases):
            if c["id"] in done:
                continue
            seed = 1000 + n + a.shard * 100000
            r = runner.once(c["case"], seed, keep=True)
            out = r.pop("_out", None)
            score, request = r.pop("score", None), r.pop("request", None)
            bad = r["outcome"] in ("crash", "violation", "timeout", "no_take", "checks_failed") or \
                  (c["expect"] == "ok" and r["outcome"] == "rejected") or \
                  (c["expect"] == "rejected" and r["outcome"] not in ("rejected", "skipped"))
            if c["expect"] == "rejected" and r["outcome"] not in ("rejected", "skipped"):
                r["signature"] = f"probe not refused ({r['outcome']}): {c['id'].split('-')[1]}"
            if bad and r["outcome"] != "skipped":
                with open(os.path.join(a.out, "failures", f"{c['id']}-{a.model}.json"), "w") as ff:
                    json.dump({"id": c["id"], "model": a.model, "case": c["case"], "seed": seed, "result": r,
                               "score": score, "request": request}, ff, default=str)
                sig = r.get("signature", r["outcome"])
                # a timeout costs the whole limit per try: not worth shrinking
                if c["expect"] == "ok" and r["outcome"] != "timeout" and shrunk.get(sig, 0) < 2:
                    shrunk[sig] = shrunk.get(sig, 0) + 1
                    r["shrunk"] = runner.shrink(c["case"], seed, sig)
            elif r["outcome"] == "ok" and c["case"]["takes"] == 1 and n % 8 == 0:
                again = runner.once(c["case"], seed)
                if again.get("outcome") == "ok" and not same_result(out, again["_out"]):
                    r["findings"].append(("warning", "nondeterministic", "same seed, different notes"))
            rec = {"id": c["id"], "model": a.model, "expect": c["expect"], "case": c["case"], "seed": seed, **r}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[{n + 1}/{len(cases)}] {c['id']} {r['outcome']} {r.get('seconds', '')}s {r.get('signature', '')}"[:240], flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
