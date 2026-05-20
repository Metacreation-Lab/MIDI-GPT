"""Extensive randomized generation benchmark.

Randomizes per scenario:
  - file, window position+size (=model_dim), mode (infill | AR-continuation)
  - number of gen tracks (2..16, bounded by file)
  - bars_per_step (1,2,4,8), tracks_per_step (1,2,4)

Reports per-scenario timings, top-N orig-wins, and aggregated speedup
breakdowns by (bars_per_step, tracks_per_step, model_dim, mode).

Set MIDIGPT_BENCH_N to override scenario count (default 20). MIDIGPT_BENCH_SEED
overrides RNG seed (default 42).
"""
import json
import os
import random
import statistics
import time
from copy import deepcopy

import pytest

from .conftest import silence_stdio, midi_files, REPO_ROOT


pytestmark = pytest.mark.benchmark

N_SCENARIOS    = int(os.environ.get("MIDIGPT_BENCH_N", 20))
SEED           = int(os.environ.get("MIDIGPT_BENCH_SEED", 42))
N_REPEATS      = 2
MODEL_DIMS     = [4, 8]  # yellow model vocab: only 4 and 8 are valid NumBars tokens
BARS_PER_STEP  = [1, 2, 4, 8]
TRACKS_PER_STEP = [1, 2, 4]
MIN_TRACKS     = 2
MAX_TRACKS     = 16
SCENARIO_RETRIES = 80


def test_full_generation_speed_extensive(yellow_orig_encoder, yellow_ref_components, capsys):
    import midigpt_legacy as midigpt
    import midigpt._core as _core
    from midigpt._converters import from_cpp
    from midigpt.inference.engine import InferenceEngine
    from midigpt.inference.config import (
        GenerationRequest, TrackPrompt, SamplingConfig,
    )
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.attributes import AttributeAnalyzer

    cfg, vocab, _, _ = yellow_ref_components

    model_path = str(REPO_ROOT / "models" / "yellow.pt")
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found: {model_path}")
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    _model = torch.jit.load(model_path, map_location="cpu")
    _model.eval()
    analyzer = AttributeAnalyzer.from_config(cfg)
    tokenizer_ = Tokenizer(cfg, analyzer)
    engine = InferenceEngine(model=_model, tokenizer=tokenizer_, analyzer=analyzer)
    engine.warmup()

    rng = random.Random(SEED)
    all_files = list(midi_files())

    # --- scenario construction ---

    def _build_scenario():
        for _attempt in range(SCENARIO_RETRIES):
            mp = rng.choice(all_files)
            try:
                with silence_stdio():
                    score = from_cpp(_core.MidiReader(cfg.resolution).read(str(mp)))
            except Exception:
                continue
            n_file_tracks = len(score.tracks)
            if n_file_tracks < MIN_TRACKS:
                continue
            max_bars = min((len(t.bars) for t in score.tracks), default=0)
            valid_dims = [d for d in MODEL_DIMS if d <= max_bars]
            if not valid_dims:
                continue
            model_dim = rng.choice(valid_dims)

            mode = rng.choice(["infill", "ar"])
            max_start = max_bars - model_dim
            win_start = rng.randint(0, max_start)
            window_bars = list(range(win_start, win_start + model_dim))

            n_gen_tracks_max = min(n_file_tracks, MAX_TRACKS)
            if n_gen_tracks_max < MIN_TRACKS:
                continue
            n_gen_tracks = rng.randint(MIN_TRACKS, n_gen_tracks_max)

            # Score tracks by note density in window — pick top n_gen_tracks
            scored = []
            for tid, t in enumerate(score.tracks):
                notes = sum(len(t.bars[b].notes) for b in window_bars)
                scored.append((tid, notes))
            scored.sort(key=lambda x: -x[1])
            top = scored[:n_gen_tracks]
            # All selected tracks must have ≥ 1 note in window (else skip scenario)
            if any(n == 0 for _, n in top):
                continue
            gen_track_ids = sorted(tid for tid, _ in top)

            # gen_bars: half the window. infill = middle, ar = trailing.
            n_gen_bars = max(1, model_dim // 2)
            if mode == "infill":
                start = (model_dim - n_gen_bars) // 2
            else:
                start = model_dim - n_gen_bars
            gen_bars_local = list(range(start, start + n_gen_bars))

            valid_bps = [b for b in BARS_PER_STEP if b <= n_gen_bars]
            if not valid_bps:
                continue
            bars_per_step = rng.choice(valid_bps)

            valid_tps = [t for t in TRACKS_PER_STEP if t <= n_gen_tracks]
            tracks_per_step = rng.choice(valid_tps)

            return dict(
                file=mp, score=score, window_bars=window_bars,
                gen_bars_local=gen_bars_local, gen_track_ids=gen_track_ids,
                mode=mode, model_dim=model_dim,
                bars_per_step=bars_per_step, tracks_per_step=tracks_per_step,
                n_file_tracks=n_file_tracks,
            )
        return None

    scenarios = []
    for _ in range(N_SCENARIOS):
        s = _build_scenario()
        if s is not None:
            scenarios.append(s)

    # --- refactored runner ---
    def _run_ref(s):
        score_prompt = deepcopy(s["score"])
        w = s["window_bars"]
        for t in score_prompt.tracks:
            t.bars = t.bars[w[0]:w[-1]+1]

        tracks = [
            TrackPrompt(id=tid, bars=s["gen_bars_local"],
                        autoregressive=(s["mode"] == "ar"))
            for tid in s["gen_track_ids"]
        ]
        request = GenerationRequest(
            tracks=tracks,
            config=SamplingConfig(
                max_attempts=1,
                silence_check=False,
                novelty_check=False,
                bars_per_step=s["bars_per_step"],
                tracks_per_step=s["tracks_per_step"],
                model_dim=s["model_dim"],
            ),
        )
        session = engine.session(score_prompt, request)
        session.enable_profiling = True
        result = session.run()
        return result, session

    # --- orig runner ---
    def _run_orig(s, piece_json_win):
        status = json.loads(midigpt.status_from_piece(piece_json_win))
        gen_set = set(s["gen_track_ids"])
        gen_bars_local = set(s["gen_bars_local"])
        for ti, t in enumerate(status["tracks"]):
            sel = t.get("selectedBars", [])
            for bi in range(len(sel)):
                sel[bi] = (ti in gen_set and bi in gen_bars_local)
            if s["mode"] == "ar" and ti in gen_set:
                t["suffix_autoregressive"] = True
        param = {
            "ckpt": model_path,
            "barsPerStep": s["bars_per_step"],
            "modelDim": s["model_dim"],
            "tracksPerStep": s["tracks_per_step"],
            "temperature": 1.0,
            "batchSize": 1,
            "percentage": 100,
            "polyphonyHardLimit": 10,
        }
        result_str, _attempts, timings_json = midigpt.sample_multi_step_timed(
            piece_json_win, json.dumps(status), json.dumps(param), 1, None,
        )
        return json.loads(result_str), json.loads(timings_json)

    # --- execute ---
    rows = []
    errors = []
    for i, s in enumerate(scenarios):
        rec = dict(
            idx=i, file=s["file"].name, mode=s["mode"],
            n_file_tracks=s["n_file_tracks"],
            n_gen_tracks=len(s["gen_track_ids"]),
            model_dim=s["model_dim"],
            n_gen_bars=len(s["gen_bars_local"]),
            bars_per_step=s["bars_per_step"],
            tracks_per_step=s["tracks_per_step"],
        )

        # refactored
        try:
            _run_ref(s)  # warm
            ref_times, ref_toks = [], 0
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                _result, session = _run_ref(s)
                ref_times.append(time.perf_counter() - t0)
                ref_toks = session.gen_count
            rec["ref_ms"]  = statistics.median(ref_times) * 1000
            rec["ref_tok"] = ref_toks
        except Exception as e:
            errors.append(f"[{i}] ref {s['file'].name} {s['mode']}: {type(e).__name__}: {e}")
            rec["ref_ms"] = None; rec["ref_tok"] = None

        # orig
        try:
            with silence_stdio():
                piece_json = yellow_orig_encoder.midi_to_json(str(s["file"]))
            piece = json.loads(piece_json)
            for t in piece.get("tracks", []):
                t["bars"] = t["bars"][s["window_bars"][0]: s["window_bars"][-1]+1]
            piece_json_win = json.dumps(piece)
            with silence_stdio():
                _run_orig(s, piece_json_win)
            orig_times, orig_toks = [], 0
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                with silence_stdio():
                    _r, tmg = _run_orig(s, piece_json_win)
                orig_times.append(time.perf_counter() - t0)
                orig_toks = int(tmg.get("tokens_generated", 0))
            rec["orig_ms"]  = statistics.median(orig_times) * 1000
            rec["orig_tok"] = orig_toks
        except Exception as e:
            errors.append(f"[{i}] orig {s['file'].name} {s['mode']}: {type(e).__name__}: {e}")
            rec["orig_ms"] = None; rec["orig_tok"] = None

        if rec["orig_ms"] and rec["ref_ms"]:
            rec["speedup"] = rec["orig_ms"] / rec["ref_ms"]
            rec["orig_tok_s"] = rec["orig_tok"] / (rec["orig_ms"]/1000) if rec["orig_tok"] else 0
            rec["ref_tok_s"]  = rec["ref_tok"]  / (rec["ref_ms"] /1000) if rec["ref_tok"]  else 0
        else:
            rec["speedup"] = None
            rec["orig_tok_s"] = rec["ref_tok_s"] = None

        rows.append(rec)

    # --- reporting ---
    def _fmt_ms(v):
        return f"{v:8.1f}ms" if v is not None else "     n/a"
    def _fmt_sp(v):
        return f"{v:6.2f}x" if v is not None else "   n/a"
    def _fmt_ts(v):
        return f"{v:7.1f}" if v is not None else "  n/a"

    with capsys.disabled():
        if errors:
            print(f"\n--- errors ({len(errors)}) ---")
            for e in errors[:20]:
                print(e)

        print("\n" + "=" * 180)
        print(f"EXTENSIVE GENERATION BENCHMARK  —  {len(rows)} scenarios "
              f"(seed={SEED}, repeats={N_REPEATS})")
        print("=" * 180)
        hdr = (f"{'#':>3} {'file':<32} {'mode':<6} "
               f"{'nT/F':<5} {'gT':>3} {'mD':>3} {'gB':>3} "
               f"{'b/s':>3} {'t/s':>3} "
               f"{'orig_ms':>10} {'ref_ms':>10} "
               f"{'o_tok/s':>9} {'r_tok/s':>9} {'speedup':>8}")
        print(hdr)
        print("-" * 180)
        for r in rows:
            print(f"{r['idx']:>3} {r['file'][:31]:<32} {r['mode']:<6} "
                  f"{r['n_gen_tracks']}/{r['n_file_tracks']:<3} "
                  f"{r['n_gen_tracks']:>3} {r['model_dim']:>3} {r['n_gen_bars']:>3} "
                  f"{r['bars_per_step']:>3} {r['tracks_per_step']:>3} "
                  f"{_fmt_ms(r['orig_ms']):>10} {_fmt_ms(r['ref_ms']):>10} "
                  f"{_fmt_ts(r['orig_tok_s']):>9} {_fmt_ts(r['ref_tok_s']):>9} "
                  f"{_fmt_sp(r['speedup']):>8}")
        print("=" * 180)
        print("COLUMNS: nT/F=gen_tracks/file_tracks  gT=gen_tracks  mD=model_dim  gB=gen_bars  "
              "b/s=bars_per_step  t/s=tracks_per_step")

        # ---- Top-N orig wins ----
        valid = [r for r in rows if r["speedup"] is not None]
        orig_wins = sorted([r for r in valid if r["speedup"] < 1], key=lambda r: r["speedup"])
        print(f"\nTOP {min(10, len(orig_wins))} SCENARIOS WHERE ORIG WAS FASTER:")
        if not orig_wins:
            print("  (none — refactored faster on every scenario)")
        else:
            print(f"  {'#':>3} {'file':<32} {'mode':<6} {'gT':>3} {'mD':>3} {'gB':>3} "
                  f"{'b/s':>3} {'t/s':>3} {'orig_ms':>10} {'ref_ms':>10} {'speedup':>8}")
            for r in orig_wins[:10]:
                print(f"  {r['idx']:>3} {r['file'][:31]:<32} {r['mode']:<6} "
                      f"{r['n_gen_tracks']:>3} {r['model_dim']:>3} {r['n_gen_bars']:>3} "
                      f"{r['bars_per_step']:>3} {r['tracks_per_step']:>3} "
                      f"{_fmt_ms(r['orig_ms'])} {_fmt_ms(r['ref_ms'])} {_fmt_sp(r['speedup'])}")

        # ---- Aggregated breakdowns ----
        def _agg_by(key):
            buckets = {}
            for r in valid:
                buckets.setdefault(r[key], []).append(r["speedup"])
            return [(k, len(v), statistics.median(v), min(v), max(v))
                    for k, v in sorted(buckets.items(), key=lambda x: str(x[0]))]

        for label, key in [("bars_per_step", "bars_per_step"),
                           ("tracks_per_step", "tracks_per_step"),
                           ("model_dim", "model_dim"),
                           ("mode", "mode"),
                           ("n_gen_tracks", "n_gen_tracks")]:
            print(f"\nSPEEDUP BREAKDOWN BY {label}:")
            print(f"  {label:<18} {'count':>6} {'median':>10} {'min':>10} {'max':>10}")
            for k, n, med, lo, hi in _agg_by(key):
                print(f"  {str(k):<18} {n:>6} {med:>9.2f}x {lo:>9.2f}x {hi:>9.2f}x")

        print()

    assert valid, "no scenarios produced valid timings"
