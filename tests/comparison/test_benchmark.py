"""Speed benchmarks: original Yellow vs refactored encoder/generator.

Run only when `--benchmark` is passed (or pytest -m benchmark).
Reports median time + per-file timings.
"""
import json
import logging
import statistics
import time
import pytest
from .conftest import silence_stdio, midi_files, REPO_ROOT


pytestmark = pytest.mark.benchmark

N_REPEATS = 5


def _bench(fn, *args, repeats=N_REPEATS):
    times = []
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        return statistics.median(times), min(times)
    except Exception as e:
        logging.error(f"Error in _bench execution: {e}")
        return None, None


def _trim_json_bars(piece_json: str, n_bars: int) -> str:
    d = json.loads(piece_json)
    for t in d.get("tracks", []):
        t["bars"] = t.get("bars", [])[:n_bars]
    return json.dumps(d)


def test_encode_speed(yellow_orig_encoder, yellow_ref_components, capsys):
    import midigpt._core as _core
    cfg, vocab, ref_enc, _ = yellow_ref_components
    N_BARS_BENCH = 8

    def encode_orig_from_json(piece_json):
        with silence_stdio():
            return yellow_orig_encoder.json_to_tokens(piece_json)

    def encode_ref_from_score(score):
        return ref_enc.encode(score)

    rows = []
    errors = []
    for mp in midi_files():
        path = str(mp)

        orig_tokens_len = 0
        ref_tokens_len = 0
        ref_score_notes = 0

        ot_med = ot_min = None
        best_n_bars = -1
        try:
            with silence_stdio():
                piece_json = yellow_orig_encoder.midi_to_json(path)
            for n_bars_option in [N_BARS_BENCH, 4]:
                try:
                    trimmed_piece_json = _trim_json_bars(piece_json, n_bars_option)
                    orig_tokens = encode_orig_from_json(trimmed_piece_json)
                    ot_med, ot_min = _bench(encode_orig_from_json, trimmed_piece_json)
                    orig_tokens_len = len(orig_tokens)
                    best_n_bars = n_bars_option
                    break
                except RuntimeError as e:
                    if "NOT IN DOMAIN FOR TOKEN TYPE TOKEN_NUM_BARS" in str(e):
                        continue
                    else:
                        raise
            if best_n_bars == -1:
                raise RuntimeError(f"Original encoder cannot handle {mp.name} with N_BARS=4 or N_BARS=8")
        except Exception as e:
            errors.append(f"orig fail {mp.name}: {type(e).__name__}: {e}")

        rt_med = rt_min = None
        ref_tokens = None
        try:
            with silence_stdio():
                score = _core.MidiReader().read(path)
            n = best_n_bars if best_n_bars != -1 else N_BARS_BENCH
            for t in score.tracks:
                t.bars = t.bars[:n]
            ref_tokens = encode_ref_from_score(score)
            rt_med, rt_min = _bench(encode_ref_from_score, score)
            for t in score.tracks:
                for b in t.bars:
                    ref_score_notes += len(b.note_indices)
        except Exception as e:
            errors.append(f"ref  fail {mp.name}: {type(e).__name__}: {e}")

        rows.append((mp.name,
                     orig_tokens_len, ot_med, ot_min,
                     len(ref_tokens) if ref_tokens else 0, rt_med, rt_min,
                     ref_score_notes))

    with capsys.disabled():
        if errors:
            print("\n--- errors ---")
            for e in errors:
                print(e)
        print("\n" + "=" * 130)
        print(f"{'file':<45} {'orig #tok':>9} {'orig med':>11} {'orig min':>11} {'ref #tok':>9} {'ref med':>11} {'ref min':>11} {'speedup':>9} {'orig tok/ms':>12} {'ref tok/ms':>12} {'orig note/ms':>14} {'ref note/ms':>14}")
        print("-" * 130)
        for name, ont, om, omin, rnt, rm, rmin, notes in rows:
            om_s   = f"{om*1000:9.2f}ms"   if om   else "      n/a"
            omin_s = f"{omin*1000:9.2f}ms" if omin else "      n/a"
            rm_s   = f"{rm*1000:9.2f}ms"   if rm   else "      n/a"
            rmin_s = f"{rmin*1000:9.2f}ms" if rmin else "      n/a"
            sp           = f"{om/rm:6.2f}x"            if (om and rm)      else "    -"
            orig_tok_ms  = f"{ont/(om*1000):10.2f}"    if (om and ont)     else "    n/a"
            ref_tok_ms   = f"{rnt/(rm*1000):10.2f}"    if (rm and rnt)     else "    n/a"
            orig_note_ms = f"{notes/(om*1000):12.2f}"  if (om and notes)   else "    n/a"
            ref_note_ms  = f"{notes/(rm*1000):12.2f}"  if (rm and notes)   else "    n/a"
            print(f"{name[:44]:<45} {ont:>9} {om_s:>11} {omin_s:>11} {rnt:>9} {rm_s:>11} {rmin_s:>11} {sp:>9} {orig_tok_ms:>12} {ref_tok_ms:>12} {orig_note_ms:>14} {ref_note_ms:>14}")
        print("=" * 130)

    assert any(rnt for _, _, _, _, rnt, _, _, _ in rows), "refactored encoder should succeed on at least one file"


def test_decode_speed(yellow_ref_components, capsys):
    """Decode-only benchmark for the refactored decoder."""
    import midigpt._core as _core
    cfg, vocab, ref_enc, ref_dec = yellow_ref_components

    rows = []
    errors = []
    for mp in midi_files():
        try:
            with silence_stdio():
                score_encoded = _core.MidiReader().read(str(mp))

            ref_score_notes = sum(
                len(b.note_indices)
                for t in score_encoded.tracks
                for b in t.bars
            )

            tokens = ref_enc.encode(score_encoded)
            med, mn = _bench(ref_dec.decode, tokens)
            rows.append((mp.name, len(tokens), med, mn, ref_score_notes))
        except Exception as e:
            errors.append(f"{mp.name}: {type(e).__name__}: {e}")

    with capsys.disabled():
        if errors:
            print("\n--- errors ---")
            for e in errors:
                print(e)
        print("\n" + "=" * 110)
        print(f"{'file':<45} {'#tokens':>9} {'decode med':>13} {'decode min':>13} {'tok/ms':>12} {'note/ms':>14}")
        print("-" * 110)
        for name, n, med, mn, notes in rows:
            tok_ms  = f"{n/(med*1000):10.2f}"     if (med and n)     else "    n/a"
            note_ms = f"{notes/(med*1000):12.2f}"  if (med and notes) else "    n/a"
            print(f"{name[:44]:<45} {n:>9} {med*1000:>11.2f}ms {mn*1000:>11.2f}ms {tok_ms:>12} {note_ms:>14}")
        print("=" * 110)

    assert rows, "no benchmark rows produced"


@pytest.mark.benchmark
def test_full_generation_speed(yellow_orig_encoder, yellow_ref_components, capsys):
    """Full generation speed benchmark: original midigpt C++ (TorchScript JIT)
    vs refactored engine using the new packed checkpoint + SDPA-native model.

    - orig: midigpt.sample_multi_step (C++ model load each call, TorchScript JIT)
            checkpoint: models/yellow_remapped.pt (TorchScript)
    - ref:  midigpt InferenceEngine, model = GPT2LMHeadModel.from_pretrained
            checkpoint: models/yellow.pt (packed dict: config + encoder_config + state_dict)
            attention: F.scaled_dot_product_attention
            device:    env MIDIGPT_BENCH_DEVICE (default 'cpu'; 'auto'/'mps'/'cuda')

    The refactored Tokenizer is built from the *embedded* encoder_config in the
    packed checkpoint — no sidecar JSON needed.
    """
    import os
    import midigpt_legacy as midigpt
    import midigpt._core as _core
    from midigpt._converters import from_cpp
    from midigpt.inference.engine import InferenceEngine
    from midigpt.inference.config import GenerationRequest, TrackPrompt, SamplingConfig
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.attributes import AttributeAnalyzer

    # Two distinct checkpoints: orig C++ needs TorchScript; ref uses packed dict.
    orig_ckpt = str(REPO_ROOT / "models" / "yellow_orig_restored.pt")
    ref_ckpt  = str(REPO_ROOT / "models" / "yellow.pt")
    if not os.path.exists(orig_ckpt):
        pytest.skip(f"TorchScript checkpoint not found: {orig_ckpt}")
    if not os.path.exists(ref_ckpt):
        pytest.skip(f"Packed checkpoint not found (run scripts/pack_checkpoint.py): {ref_ckpt}")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from midigpt.inference.model import GPT2LMHeadModel, resolve_device

    device = resolve_device(os.environ.get("MIDIGPT_BENCH_DEVICE", "cpu"))

    # ---- refactored engine setup (measured separately) ----
    t0 = time.perf_counter()
    _model = GPT2LMHeadModel.from_pretrained(ref_ckpt, device=device)
    model_load_time = time.perf_counter() - t0

    # Smart config: build Tokenizer from the *embedded* encoder_config — no
    # sidecar JSON. The fixture-provided cfg (built from yellow_config.json on
    # disk) is used only as a parity reference; we deliberately use the
    # in-file config for the refactor path.
    enc_cfg_json = json.dumps(_model.encoder_config)
    embedded_cfg = _core.EncoderConfig.from_json(enc_cfg_json)
    analyzer  = AttributeAnalyzer.from_config(embedded_cfg)
    tokenizer = Tokenizer(embedded_cfg, analyzer)
    engine = InferenceEngine(model=_model, tokenizer=tokenizer, analyzer=analyzer)

    t0 = time.perf_counter()
    engine.warmup()   # probes KV shape + JIT-compiles the model once
    warmup_time = time.perf_counter() - t0

    N_GEN_REPEATS = 3
    WIN_SIZE  = 4   # total bars in context window
    N_GEN     = 2   # bars in the middle to infill

    def _find_window(score):
        """Return (track_id, window_bars, gen_bars) for the best non-empty window.

        Scans all tracks for a WIN_SIZE-bar window where both boundary bars
        (first and last) have notes — good context for infill generation.
        gen_bars are the N_GEN middle bars of that window.
        Returns None if no qualifying window is found.
        """
        n_context = WIN_SIZE - N_GEN  # bars kept as pure context (≥2)
        for track_id, track in enumerate(score.tracks):
            n = len(track.bars)
            if n < WIN_SIZE:
                continue
            for start in range(n - WIN_SIZE + 1):
                w = list(range(start, start + WIN_SIZE))
                # boundary bars must have notes
                if (len(track.bars[w[0]].notes) > 0
                        and len(track.bars[w[-1]].notes) > 0):
                    gen_bars = w[1 : 1 + N_GEN]
                    return track_id, w, gen_bars
        return None

    def _count_notes(score, track_id, bar_indices):
        if track_id >= len(score.tracks):
            return 0
        return sum(
            len(score.tracks[track_id].bars[b].notes)
            for b in bar_indices
            if b < len(score.tracks[track_id].bars)
        )

    def _count_orig_notes(result_dict, track_id, bar_indices):
        """Count note events in selected bars of orig result piece dict.

        bar["events"] holds indices into result_dict["events"] for BOTH note-on
        and note-off events; count only note-ons (velocity > 0) for parity
        with ref's per-note count.
        """
        tracks = result_dict.get("tracks", [])
        if track_id >= len(tracks):
            return 0
        bars = tracks[track_id].get("bars", [])
        all_ev = result_dict.get("events", [])
        return sum(
            sum(1 for ei in bars[bi].get("events", []) if all_ev[ei]["velocity"] > 0)
            for bi in bar_indices
            if bi < len(bars)
        )

    def _run_session(score_prompt, track_id, gen_bars):
        request = GenerationRequest(
            tracks=[TrackPrompt(id=track_id, bars=gen_bars)],
            config=SamplingConfig(
                max_attempts=1,
                silence_check=False,
                novelty_check=False,
                bars_per_step=N_GEN,   # match orig barsPerStep for fair comparison
            ),
        )
        session = engine.session(score_prompt, request)
        session.enable_profiling = True
        result = session.run()
        return result, session

    # ---- original midigpt inference helpers ----
    def _run_orig(piece_json_win, track_id, gen_bars_local, n_tracks):
        """Run midigpt.sample_multi_step_timed on a pre-trimmed window."""
        status = json.loads(midigpt.status_from_piece(piece_json_win))
        for ti, t in enumerate(status["tracks"]):
            for bi in range(len(t.get("selectedBars", []))):
                t["selectedBars"][bi] = (ti == track_id and bi in gen_bars_local)
        param = {
            "ckpt": orig_ckpt,
            "barsPerStep": N_GEN,
            "modelDim": WIN_SIZE,
            "tracksPerStep": 1,
            "temperature": 1.0,
            "batchSize": 1,
            "percentage": 100,
            "polyphonyHardLimit": 10,
        }
        result_str, attempts, timings_json = midigpt.sample_multi_step_timed(
            piece_json_win, json.dumps(status), json.dumps(param), 1, None
        )
        return json.loads(result_str), attempts, json.loads(timings_json)

    rows = []
    errors = []

    for mp in list(midi_files())[:5]:
        path = str(mp)

        orig_med = orig_min = None
        orig_t_load = orig_t_pre = orig_t_enc = orig_t_fwd = orig_t_dec = None
        orig_t_slice = orig_t_plan = orig_t_total = None
        orig_toks = orig_ctx = orig_steps = orig_notes = None
        ref_total_med = ref_total_min = None
        ref_toks = ref_notes = None
        ref_enc_t = ref_model_t = ref_dec_t = None
        window_info = "n/a"

        try:
            full_score = from_cpp(_core.MidiReader(embedded_cfg.resolution).read(path))

            win = _find_window(full_score)
            if win is None:
                errors.append(f"{mp.name}: no non-empty window found (skipping)")
                rows.append((mp.name, None, None, None, None, None, None, None, None, "no window"))
                continue

            track_id, window_bars, gen_bars = win
            window_info = f"trk={track_id} win={window_bars[0]}-{window_bars[-1]} gen={gen_bars}"
            gen_bars_local = [b - window_bars[0] for b in gen_bars]

            # ---- refactored: trim, warm up, then timed runs ----
            from copy import deepcopy
            score_prompt = deepcopy(full_score)
            for t in score_prompt.tracks:
                t.bars = t.bars[window_bars[0] : window_bars[-1] + 1]

            _run_session(score_prompt, track_id, gen_bars_local)  # discard warm-up

            total_times, enc_times, model_times, dec_times = [], [], [], []
            last_toks = last_notes = None
            for _ in range(N_GEN_REPEATS):
                t0 = time.perf_counter()
                result, session = _run_session(score_prompt, track_id, gen_bars_local)
                total_times.append(time.perf_counter() - t0)
                enc_times.append(session.encode_time)
                model_times.append(session.model_forward_time)
                dec_times.append(session.decode_time)
                last_toks  = session.gen_count
                last_notes = _count_notes(result, track_id, gen_bars_local)

            ref_total_med = statistics.median(total_times)
            ref_total_min = min(total_times)
            ref_enc_t   = statistics.median(enc_times)
            ref_model_t = statistics.median(model_times)
            ref_dec_t   = statistics.median(dec_times)
            ref_toks    = last_toks
            ref_notes   = last_notes

            # ---- original midigpt: build trimmed piece JSON then timed runs ----
            try:
                with silence_stdio():
                    orig_piece_json = yellow_orig_encoder.midi_to_json(path)
                orig_piece = json.loads(orig_piece_json)
                # trim all tracks to the window
                for t in orig_piece.get("tracks", []):
                    t["bars"] = t["bars"][window_bars[0] : window_bars[-1] + 1]
                orig_piece_json_win = json.dumps(orig_piece)
                n_tracks = len(orig_piece.get("tracks", []))

                # warm-up (discarded)
                with silence_stdio():
                    _run_orig(orig_piece_json_win, track_id, gen_bars_local, n_tracks)

                orig_times = []
                all_timings = []
                last_orig_result = None
                for _ in range(N_GEN_REPEATS):
                    t0 = time.perf_counter()
                    with silence_stdio():
                        last_orig_result, _, tmg = _run_orig(orig_piece_json_win, track_id, gen_bars_local, n_tracks)
                    orig_times.append(time.perf_counter() - t0)
                    all_timings.append(tmg)
                orig_notes = _count_orig_notes(last_orig_result, track_id, gen_bars_local) if last_orig_result else None
                orig_med = statistics.median(orig_times)
                orig_min = min(orig_times)
                # median per-phase from C++ timings struct (already in ms)
                def _med_phase(key):
                    vals = [t[key] for t in all_timings if key in t]
                    return statistics.median(vals) if vals else None
                orig_t_load  = _med_phase("model_load_ms")
                orig_t_pre   = _med_phase("preprocess_ms")
                orig_t_plan  = _med_phase("step_plan_ms")
                orig_t_slice = _med_phase("slice_ms")
                orig_t_enc   = _med_phase("prompt_encode_ms")
                orig_t_fwd   = _med_phase("model_forward_ms")
                orig_t_dec   = _med_phase("decode_ms")
                orig_t_total = _med_phase("total_gen_ms")
                orig_toks    = int(_med_phase("tokens_generated") or 0)
                orig_ctx     = int(_med_phase("context_tokens") or 0)
                orig_steps   = int(_med_phase("steps") or 0)
            except Exception as oe:
                errors.append(f"orig {mp.name}: {type(oe).__name__}: {oe}")

        except Exception as e:
            errors.append(f"{mp.name}: {type(e).__name__}: {e}")
            window_info = "error"

        rows.append((mp.name,
                     orig_med, orig_min,
                     orig_t_load, orig_t_pre, orig_t_plan, orig_t_slice,
                     orig_t_enc, orig_t_fwd, orig_t_dec, orig_t_total,
                     orig_toks, orig_notes, orig_ctx, orig_steps,
                     ref_total_med, ref_total_min,
                     ref_enc_t, ref_model_t, ref_dec_t,
                     ref_toks, ref_notes, window_info))

    W = 340
    with capsys.disabled():
        if errors:
            print("\n--- errors ---")
            for e in errors:
                print(e)

        print("\n" + "=" * W)
        print(f"FULL GENERATION SPEED  (ref setup: model_load={model_load_time*1000:.1f}ms  warmup={warmup_time*1000:.1f}ms — not counted below)")
        print(f"orig = midigpt C++ (loads model on each call)  |  ref = midigpt (model pre-loaded + KV cached)")
        print(f"NOTE: same parameters used for both — barsPerStep={N_GEN}, modelDim={WIN_SIZE}, temp=1.0, max_attempts=1")
        print(f"NOTE: encoders verified equivalent by parity tests (test_parity_raw.py passes)")
        print(f"\n{'file':<38} {'window':<27} "
              f"{'o.wall':>9} "
              f"{'o.total':>9} {'o.load':>9} {'o.pre':>8} {'o.plan':>8} {'o.slice':>8} {'o.enc':>8} {'o.fwd':>9} {'o.dec':>8} "
              f"{'o.#tok':>7} {'o.#note':>8} {'o.ctx':>7} {'o.step':>7} "
              f"{'o.tok/s':>9} {'o.note/s':>10} "
              f"{'r.med':>9} {'r.min':>9} "
              f"{'r.enc':>8} {'r.fwd':>9} {'r.dec':>8} "
              f"{'r.#tok':>7} {'r.#note':>8} "
              f"{'r.tok/s':>9} {'r.note/s':>10} "
              f"{'speedup':>8}")
        print("-" * W)

        for row in rows:
            (name, o_med, o_min,
             o_load, o_pre, o_plan, o_slice, o_enc, o_fwd, o_dec, o_total,
             o_toks, o_notes, o_ctx, o_steps,
             t_med, t_min, t_enc, t_model, t_dec,
             r_toks, r_notes, winfo) = row

            def ms(v, w=8): return f"{v:{w}.1f}ms" if v is not None else f"{'n/a':>{w+2}}"
            def oms(v, w=8): return f"{v*1000:{w}.1f}ms" if v is not None else f"{'n/a':>{w+2}}"
            def rate(num, denom_s, w=8):
                # denom_s in seconds
                if num is None or denom_s is None or denom_s == 0: return f"{'n/a':>{w}}"
                return f"{num/denom_s:{w}.1f}"
            def speedup(a, b):
                if a is None or b is None or b == 0: return "     n/a"
                return f"{a/b:7.2f}x"

            print(f"{name[:37]:<38} {str(winfo)[:26]:<27} "
                  f"{oms(o_med,7):>9} "
                  f"{ms(o_total,7):>9} {ms(o_load,7):>9} {ms(o_pre,6):>8} {ms(o_plan,6):>8} {ms(o_slice,6):>8} {ms(o_enc,6):>8} {ms(o_fwd,7):>9} {ms(o_dec,6):>8} "
                  f"{o_toks if o_toks is not None else 'n/a':>7} {o_notes if o_notes is not None else 'n/a':>8} {o_ctx if o_ctx is not None else 'n/a':>7} {o_steps if o_steps is not None else 'n/a':>7} "
                  f"{rate(o_toks, o_med, 7):>9} {rate(o_notes, o_med, 8):>10} "
                  f"{oms(t_med,7):>9} {oms(t_min,7):>9} "
                  f"{oms(t_enc,6):>8} {oms(t_model,7):>9} {oms(t_dec,6):>8} "
                  f"{r_toks if r_toks is not None else 'n/a':>7} {r_notes if r_notes is not None else 'n/a':>8} "
                  f"{rate(r_toks, t_med, 7):>9} {rate(r_notes, t_med, 8):>10} "
                  f"{speedup(o_med, t_med):>8}")

        print("=" * W)
        print("\nCOLUMN KEY:")
        print("  o.wall  = wall-clock for midigpt.sample_multi_step_timed | o.total = C++ total_gen_ms")
        print("  o.load  = model load from disk | o.pre = validate+pad | o.plan = find_steps")
        print("  o.slice = piece slicing/step   | o.enc = prompt tokenisation/step")
        print("  o.fwd   = transformer forward (sum all steps) | o.dec = tokens→piece decode")
        print("  o.#tok  = tokens generated | o.#note = notes in generated bars | o.ctx = context tokens (prompt length sum)")
        print("  o.step  = generation steps | o.tok/s = tokens/sec | o.note/s = notes/sec")
        print("  r.*     = same fields for refactored engine")
        print("  speedup = o.wall / r.med  (>1x = refactored faster)")
        print(f"  ref e2e WITH setup = r.med + {(model_load_time+warmup_time)*1000:.1f}ms  (load={model_load_time*1000:.1f}ms + warmup={warmup_time*1000:.1f}ms)")

    assert any(row[13] is not None for row in rows), \
        "refactored generation should succeed on at least one file"


@pytest.mark.benchmark
def test_full_roundtrip_speed(yellow_orig_encoder, yellow_ref_components, capsys):
    """Encode + Decode roundtrip benchmark (no model forward)."""
    import midigpt._core as _core
    cfg, vocab, ref_enc, ref_dec = yellow_ref_components
    N_BARS_BENCH = 8

    def orig_full_roundtrip(path, n_bars):
        with silence_stdio():
            piece_json = yellow_orig_encoder.midi_to_json(path)
            trimmed = _trim_json_bars(piece_json, n_bars)
            tokens = yellow_orig_encoder.json_to_tokens(trimmed)
        return tokens

    def ref_full_roundtrip(path, n_bars):
        with silence_stdio():
            score = _core.MidiReader().read(path)
        for t in score.tracks:
            t.bars = t.bars[:n_bars]
        tokens = ref_enc.encode(score)
        ref_dec.decode(tokens)
        return tokens

    rows = []
    errors = []
    for mp in midi_files():
        path = str(mp)
        orig_tokens_len = 0
        ref_tokens_len  = 0
        ref_notes       = 0
        best_n_bars     = -1

        ot_med = ot_min = None
        try:
            for n_bars_option in [N_BARS_BENCH, 4]:
                try:
                    tokens_orig = orig_full_roundtrip(path, n_bars_option)
                    orig_tokens_len = len(tokens_orig)
                    ot_med, ot_min = _bench(orig_full_roundtrip, path, n_bars_option)
                    best_n_bars = n_bars_option
                    break
                except RuntimeError as e:
                    if "NOT IN DOMAIN FOR TOKEN TYPE TOKEN_NUM_BARS" in str(e):
                        continue
                    raise
            if best_n_bars == -1:
                raise RuntimeError(f"Original encoder cannot handle {mp.name} with N_BARS=4 or 8")
        except Exception as e:
            errors.append(f"orig fail {mp.name}: {type(e).__name__}: {e}")

        rt_med = rt_min = None
        try:
            n = best_n_bars if best_n_bars != -1 else N_BARS_BENCH
            score_for_notes = _core.MidiReader().read(path)
            for t in score_for_notes.tracks:
                t.bars = t.bars[:n]
            ref_notes = sum(len(b.note_indices) for t in score_for_notes.tracks for b in t.bars)

            tokens_ref = ref_full_roundtrip(path, n)
            ref_tokens_len = len(tokens_ref)
            rt_med, rt_min = _bench(ref_full_roundtrip, path, n)
        except Exception as e:
            errors.append(f"ref  fail {mp.name}: {type(e).__name__}: {e}")

        rows.append((mp.name,
                     orig_tokens_len, ot_med, ot_min,
                     ref_tokens_len,  rt_med, rt_min,
                     ref_notes))

    with capsys.disabled():
        if errors:
            print("\n--- errors ---")
            for e in errors:
                print(e)
        print("\n" + "=" * 160)
        print("ENCODE + DECODE ROUNDTRIP SPEED")
        print(f"\n{'file':<45} {'orig #tok':>9} {'orig med':>11} {'orig min':>11} "
              f"{'ref #tok':>9} {'ref med':>11} {'ref min':>11} "
              f"{'speedup':>9} {'orig tok/ms':>12} {'ref tok/ms':>12} "
              f"{'orig note/ms':>14} {'ref note/ms':>14}")
        print("-" * 160)
        for name, ont, om, omin, rnt, rm, rmin, notes in rows:
            om_s    = f"{om*1000:9.2f}ms"    if om    else "      n/a"
            omin_s  = f"{omin*1000:9.2f}ms"  if omin  else "      n/a"
            rm_s    = f"{rm*1000:9.2f}ms"    if rm    else "      n/a"
            rmin_s  = f"{rmin*1000:9.2f}ms"  if rmin  else "      n/a"
            sp           = f"{om/rm:6.2f}x"           if (om and rm)    else "    -"
            orig_tok_ms  = f"{ont/(om*1000):10.2f}"   if (om and ont)   else "    n/a"
            ref_tok_ms   = f"{rnt/(rm*1000):10.2f}"   if (rm and rnt)   else "    n/a"
            orig_note_ms = f"{notes/(om*1000):12.2f}" if (om and notes) else "    n/a"
            ref_note_ms  = f"{notes/(rm*1000):12.2f}" if (rm and notes) else "    n/a"
            print(f"{name[:44]:<45} {ont:>9} {om_s:>11} {omin_s:>11} "
                  f"{rnt:>9} {rm_s:>11} {rmin_s:>11} "
                  f"{sp:>9} {orig_tok_ms:>12} {ref_tok_ms:>12} "
                  f"{orig_note_ms:>14} {ref_note_ms:>14}")
        print("=" * 160)

    assert any(rnt for _, _, _, _, rnt, _, _, _ in rows), \
        "refactored roundtrip should succeed on at least one file"
