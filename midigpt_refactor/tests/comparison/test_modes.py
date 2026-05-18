"""Mode-specific tokenization tests: autoregressive, suffix-AR, multi-fill (infill).

Compares full token sequences between original Yellow encoder and refactored
encoder for each mode. Includes randomized fuzz tests generating many random
generation tasks (random mode, random bar selections, random controls).
"""
import random
import pytest
import json
from .conftest import silence_stdio, pretty_ref, pretty_orig, diff_report, midi_files


def _load_score(midi_path: str, n_bars: int = 4):
    import midigpt_refactor._core as _core
    with silence_stdio():
        score = _core.MidiReader().read(midi_path)
    for t in score.tracks:
        t.bars = t.bars[:n_bars]
    return score


def _find_midi(min_bars: int):
    import midigpt_refactor._core as _core
    with silence_stdio():
        for mp in midi_files():
            try:
                s = _core.MidiReader().read(str(mp))
                if max(len(t.bars) for t in s.tracks) >= min_bars:
                    return str(mp)
            except Exception:
                pass
    return None


def _get_test_midis(min_bars: int = 4, max_files: int = 1000):
    """Return list of MIDI paths with at least min_bars bars."""
    import midigpt_refactor._core as _core
    results = []
    with silence_stdio():
        for mp in midi_files():
            try:
                s = _core.MidiReader().read(str(mp))
                if max(len(t.bars) for t in s.tracks) >= min_bars:
                    results.append(str(mp))
                    if len(results) >= max_files:
                        break
            except Exception:
                pass
    return results


def _orig_encode_ar(orig_enc, midi_path, n_bars):
    """Encode with original encoder in AR mode (default)."""
    with silence_stdio():
        try:
            pj = orig_enc.midi_to_json(midi_path)
        except Exception:
            return None
    d = json.loads(pj)
    for t in d.get("tracks", []):
        t["bars"] = t.get("bars", [])[:n_bars]
    pj_trimmed = json.dumps(d)
    with silence_stdio():
        try:
            return orig_enc.json_to_tokens(pj_trimmed)
        except Exception:
            return None


def _orig_encode_multifill(orig_enc, midi_path, n_bars, fill_set):
    """Encode with original encoder in multi-fill mode."""
    cfg = orig_enc.config
    cfg.do_multi_fill = True
    cfg.multi_fill = fill_set
    with silence_stdio():
        try:
            pj = orig_enc.midi_to_json(midi_path)
        except Exception:
            cfg.do_multi_fill = False
            return None
    d = json.loads(pj)
    for t in d.get("tracks", []):
        t["bars"] = t.get("bars", [])[:n_bars]
    pj_trimmed = json.dumps(d)
    with silence_stdio():
        try:
            tokens = orig_enc.json_to_tokens(pj_trimmed)
        except Exception:
            tokens = None
    cfg.do_multi_fill = False
    return tokens


def _ref_encode(vocab, ref_enc, midi_path, n_bars, do_multi_fill=False,
                multi_fill=None, partial_track=-1, partial_bars=-1):
    """Encode with refactored encoder, optionally in multi-fill or partial mode."""
    import midigpt_refactor._core as _core
    cfg = _core.EncoderConfig.from_json(vocab.config().to_json())
    if do_multi_fill:
        cfg.supports_infill = True
    v = _core.Vocabulary(cfg)
    e = _core.Encoder(v)
    opts = _core.EncodeOptions()
    if do_multi_fill and multi_fill:
        opts.multi_fill = multi_fill
    opts.partial_encode_track_index = partial_track
    opts.partial_encode_track_bars = partial_bars
    with silence_stdio():
        score = _core.MidiReader().read(midi_path)
    for t in score.tracks:
        t.bars = t.bars[:n_bars]
    return e.encode(score, opts)


def _strip_track_header_attrs(normalized):
    """Drop all tokens that fall between a Track/Instrument header and the
    first Bar of that track. Orig and ref encoders compute *different*
    attribute sets for the track header (orig: VelocityLevel summaries; ref:
    MinPolyphony/MaxPolyphony/MinNoteDuration/MaxNoteDuration) so the header
    block is not directly comparable. Note-stream VelocityLevel tokens are
    preserved (they appear after Bar)."""
    from midigpt_refactor._core import TokenType
    HEADER_TYPES = {TokenType.Track, TokenType.Instrument}
    out = []
    in_header = False
    for tt, val in normalized:
        if tt in HEADER_TYPES:
            in_header = True
            out.append((tt, val))
            continue
        if in_header:
            if tt == TokenType.Bar:
                in_header = False
                out.append((tt, val))
            # otherwise drop
            continue
        out.append((tt, val))
    return out


def _normalize_ref_tokens(vocab, tokens):
    """Convert raw ref tokens to list of (TokenType, value) tuples.
    Normalizes metadata values to ignore minor mapping discrepancies."""
    from midigpt_refactor._core import TokenType
    METADATA_TYPES = (
        TokenType.PieceStart, TokenType.NumBars, TokenType.Track, 
        TokenType.Instrument, TokenType.TimeSig,
        TokenType.MinPolyphony, TokenType.MaxPolyphony, 
        TokenType.MinNoteDuration, TokenType.MaxNoteDuration,
        TokenType.VelocityLevel, TokenType.NoteDensity,
        TokenType.OnsetPolyphony, TokenType.PitchRange,
        TokenType.KeySignature, TokenType.NoteDurationDist,
        TokenType.Tension, TokenType.SilenceProportion,
        TokenType.PitchClassSet
    )
    out = []
    for t in tokens:
        try:
            tt, val = vocab.decode(t)
            if tt in METADATA_TYPES:
                val = 0
            out.append((tt, val))
        except Exception:
            out.append(("?", t))
    return out


def _normalize_orig_tokens(orig_enc, vocab, tokens):
    """Convert original tokens to (TokenType, value) tuples via pretty-string parsing.
    Normalizes metadata values to 0."""
    from midigpt_refactor._core import TokenType
    # Map original pretty type names to refactored TokenType
    type_map = {
        "TOKEN_PIECE_START": TokenType.PieceStart,
        "TOKEN_NUM_BARS": TokenType.NumBars,
        "TOKEN_TRACK": TokenType.Track,
        "TOKEN_INSTRUMENT": TokenType.Instrument,
        "TOKEN_BAR": TokenType.Bar,
        "TOKEN_TIME_SIGNATURE": TokenType.TimeSig,
        "TOKEN_TIME_ABSOLUTE_POS": TokenType.TimeAbsolutePos,
        "TOKEN_VELOCITY_LEVEL": TokenType.VelocityLevel,
        "TOKEN_NOTE_ONSET": TokenType.NoteOnset,
        "TOKEN_NOTE_DURATION": TokenType.NoteDuration,
        "TOKEN_BAR_END": TokenType.BarEnd,
        "TOKEN_TRACK_END": TokenType.TrackEnd,
        "TOKEN_FILL_IN_PLACEHOLDER": TokenType.FillInPlaceholder,
        "TOKEN_FILL_IN_START": TokenType.FillInStart,
        "TOKEN_FILL_IN_END": TokenType.FillInEnd,
        "TOKEN_PIECE_END": TokenType.PieceEnd,
        "TOKEN_MASK_BAR": TokenType.MaskBar,
        # Attributes
        "TOKEN_DENSITY_LEVEL": TokenType.NoteDensity,
        "TOKEN_POLYPHONY_LEVEL": TokenType.OnsetPolyphony,
        "TOKEN_PITCH_RANGE": TokenType.PitchRange,
        "TOKEN_KEY_SIGNATURE": TokenType.KeySignature,
        "TOKEN_NOTE_DURATION_DIST": TokenType.NoteDurationDist,
        "TOKEN_TENSION": TokenType.Tension,
        "TOKEN_SILENCE_PROPORTION": TokenType.SilenceProportion,
        "TOKEN_PITCH_CLASS_SET": TokenType.PitchClassSet,
        # Legacy attributes
        "TOKEN_MIN_POLYPHONY": TokenType.MinPolyphony,
        "TOKEN_MAX_POLYPHONY": TokenType.MaxPolyphony,
        "TOKEN_MIN_NOTE_DURATION": TokenType.MinNoteDuration,
        "TOKEN_MAX_NOTE_DURATION": TokenType.MaxNoteDuration,
    }
    METADATA_TYPES = (
        TokenType.PieceStart, TokenType.NumBars, TokenType.Track, 
        TokenType.Instrument, TokenType.TimeSig,
        TokenType.MinPolyphony, TokenType.MaxPolyphony, 
        TokenType.MinNoteDuration, TokenType.MaxNoteDuration,
        TokenType.VelocityLevel, TokenType.NoteDensity,
        TokenType.OnsetPolyphony, TokenType.PitchRange,
        TokenType.KeySignature, TokenType.NoteDurationDist,
        TokenType.Tension, TokenType.SilenceProportion,
        TokenType.PitchClassSet
    )
    out = []
    for t in tokens:
        p = pretty_orig(orig_enc, t)
        if " = " in p:
            t_str, v_str = p.split(" = ", 1)
        else:
            t_str, v_str = p, "0"
        
        tt = type_map.get(t_str, "?")
        try:
            val = int(v_str) if v_str.isdigit() else 0
        except Exception:
            val = 0
        
        if tt in METADATA_TYPES:
            val = 0
        out.append((tt, val))
    return out


# ---------------------------------------------------------------------------
# Autoregressive (default) mode
# ---------------------------------------------------------------------------

class TestAutoregressive:
    def test_emits_track_end(self, yellow_ref_components):
        from midigpt_refactor._core import TokenType
        cfg, vocab, ref_enc, _ = yellow_ref_components
        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = ref_enc.encode(score)
        te_lo, te_hi = vocab.range(TokenType.TrackEnd)
        assert any(te_lo <= t < te_hi for t in tokens), \
            "autoregressive encoding should emit TrackEnd"

    def test_no_fill_tokens(self, yellow_ref_components):
        from midigpt_refactor._core import TokenType
        cfg, vocab, ref_enc, _ = yellow_ref_components
        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = ref_enc.encode(score)
        for tt in (TokenType.FillInPlaceholder, TokenType.FillInStart, TokenType.FillInEnd):
            lo, hi = vocab.range(tt)
            if lo == -1:
                continue
            assert not any(lo <= t < hi for t in tokens), \
                f"unexpected {tt} token in autoregressive encoding"

    def test_full_sequence_matches_original(self, yellow_orig_encoder, yellow_ref_components):
        """Encode → remap → decode → re-encode roundtrip: re-encoded tokens
        should equal the remapped-to-ref orig tokens."""
        from midigpt_refactor.compat import build_orig_to_ref_mapping, remap_orig_tokens
        cfg, vocab, ref_enc, ref_dec = yellow_ref_components
        mapping = build_orig_to_ref_mapping(yellow_orig_encoder, vocab)
        midis = _get_test_midis(4, max_files=1000)
        if not midis:
            pytest.skip("no test MIDI available")

        compared = 0
        failures = []
        for mp in midis:
            orig = _orig_encode_ar(yellow_orig_encoder, mp, 4)
            if orig is None:
                continue
            orig = remap_orig_tokens(orig, mapping)
            # Decode with refactored decoder, then re-encode
            with silence_stdio():
                score = ref_dec.decode(orig)
            # Clear attributes — original tokens had them stripped
            for t in score.tracks:
                t.attributes.clear()
            ref = ref_enc.encode(score)

            compared += 1
            if orig != ref:
                report = diff_report(
                    "orig_remapped", orig, lambda t: pretty_ref(vocab, t),
                    "ref",           ref,  lambda t: pretty_ref(vocab, t),
                )
                failures.append((mp.split('/')[-1], report))

        if compared == 0:
            pytest.skip("no comparable MIDI files for original encoder")
        if failures:
            msg = f"\nAR roundtrip parity failed on {len(failures)}/{compared}:\n"
            for name, report in failures[:3]:
                msg += f"\n--- {name} ---\n{report}\n"
            pytest.fail(msg)


# ---------------------------------------------------------------------------
# Suffix-autoregressive mode (partial encode for live continuation)
# ---------------------------------------------------------------------------

class TestSuffixAutoregressive:
    """Realtime path: encode all but the last K bars on the agent track."""

    def test_partial_encoding_drops_tail_and_track_end(self, yellow_config_text):
        """When config.partial_encode_track_index/bars is set, encoder
        truncates that track's bars and omits TrackEnd."""
        import midigpt_refactor._core as _core
        from midigpt_refactor._core import TokenType
        cfg = _core.EncoderConfig.from_json(yellow_config_text)
        vocab = _core.Vocabulary(cfg)
        enc = _core.Encoder(vocab)
        opts = _core.EncodeOptions()
        opts.partial_encode_track_index = 0
        opts.partial_encode_track_bars = 2  # keep first 2 of 4 bars on track 0

        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = enc.encode(score, opts)

        # Count Bar tokens — should be 2 (truncated) + n_bars * (other tracks)
        bar_lo, bar_hi = vocab.range(TokenType.Bar)
        bar_count = sum(1 for t in tokens if bar_lo <= t < bar_hi)
        n_other = max(0, len(score.tracks) - 1)
        expected_min = 2
        expected_max = 2 + n_other * 4
        assert expected_min <= bar_count <= expected_max, (
            f"bar count {bar_count} not in [{expected_min},{expected_max}]"
        )

    def test_suffix_ar_no_track_end_on_partial_track(self, yellow_config_text):
        """Partial track should not have TrackEnd token."""
        import midigpt_refactor._core as _core
        from midigpt_refactor._core import TokenType
        cfg = _core.EncoderConfig.from_json(yellow_config_text)
        vocab = _core.Vocabulary(cfg)
        enc = _core.Encoder(vocab)
        opts = _core.EncodeOptions()
        opts.partial_encode_track_index = 0
        opts.partial_encode_track_bars = 2

        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = enc.encode(score, opts)

        # Count TrackEnd tokens — should be (num_tracks - 1), not num_tracks
        te_lo, te_hi = vocab.range(TokenType.TrackEnd)
        n_track_ends = sum(1 for t in tokens if te_lo <= t < te_hi)
        expected = max(0, len(score.tracks) - 1)
        assert n_track_ends == expected, (
            f"expected {expected} TrackEnd tokens, got {n_track_ends}"
        )


# ---------------------------------------------------------------------------
# Multi-fill (infill) mode
# ---------------------------------------------------------------------------

class TestMultiFill:
    """do_multi_fill=True: bars in `multi_fill` set become FillInPlaceholder
    in the main sequence; their content is moved to FillInStart/notes/FillInEnd
    blocks appended after the main sequence."""

    def _make_config_and_opts(self, yellow_config_text, fill_set):
        import midigpt_refactor._core as _core
        cfg = _core.EncoderConfig.from_json(yellow_config_text)
        cfg.supports_infill = True
        opts = _core.EncodeOptions()
        opts.multi_fill = fill_set
        return cfg, opts

    def test_placeholder_emitted_in_main_sequence(self, yellow_config_text):
        import midigpt_refactor._core as _core
        from midigpt_refactor._core import TokenType
        cfg, opts = self._make_config_and_opts(yellow_config_text, {(0, 1), (0, 2)})
        vocab = _core.Vocabulary(cfg)
        enc = _core.Encoder(vocab)
        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = enc.encode(score, opts)

        ph_lo, ph_hi = vocab.range(TokenType.FillInPlaceholder)
        fs_lo, fs_hi = vocab.range(TokenType.FillInStart)
        fe_lo, fe_hi = vocab.range(TokenType.FillInEnd)
        assert ph_lo != -1, "FillInPlaceholder must exist in vocab"

        n_placeholders = sum(1 for t in tokens if ph_lo <= t < ph_hi)
        n_starts = sum(1 for t in tokens if fs_lo <= t < fs_hi)
        n_ends = sum(1 for t in tokens if fe_lo <= t < fe_hi)

        assert n_placeholders == 2, f"expected 2 placeholders, got {n_placeholders}"
        assert n_starts == 2, f"expected 2 FillInStart tokens, got {n_starts}"
        assert n_ends == 2, f"expected 2 FillInEnd tokens, got {n_ends}"

    def test_decode_resolves_placeholders(self, yellow_config_text):
        """Decoder should reassemble placeholders with their fill blocks."""
        import midigpt_refactor._core as _core
        cfg, opts = self._make_config_and_opts(yellow_config_text, {(0, 1)})
        vocab = _core.Vocabulary(cfg)
        enc = _core.Encoder(vocab)
        dec = _core.Decoder(vocab)
        mp = _find_midi(4)
        if mp is None:
            pytest.skip("no test MIDI available")
        score = _load_score(mp, 4)
        tokens = enc.encode(score, opts)
        decoded = dec.decode(tokens)
        assert len(decoded.tracks[0].bars) == 4

    def test_full_sequence_matches_original(self, yellow_orig_encoder, yellow_ref_components):
        """Multi-fill encoding via remap: orig tokens are remapped into the ref
        vocab; the re-encoded ref tokens should match (modulo normalized
        metadata values)."""
        from midigpt_refactor.compat import build_orig_to_ref_mapping, remap_orig_tokens
        cfg, vocab, ref_enc, ref_dec = yellow_ref_components
        mapping = build_orig_to_ref_mapping(yellow_orig_encoder, vocab)
        midis = _get_test_midis(4, max_files=1000)
        if not midis:
            pytest.skip("no test MIDI available")

        import midigpt_refactor._core as _core
        compared = 0
        failures = []
        for mp in midis:
            fill_set = {(0, 1), (0, 2)}
            orig = _orig_encode_multifill(yellow_orig_encoder, mp, 4, fill_set)
            if orig is None:
                continue
            # Decode original AR tokens to get a Score, then re-encode with multi-fill
            ar_tokens = _orig_encode_ar(yellow_orig_encoder, mp, 4)
            if ar_tokens is None:
                continue
            ar_tokens_remapped = remap_orig_tokens(ar_tokens, mapping)
            with silence_stdio():
                score = ref_dec.decode(ar_tokens_remapped)
            # Clear attributes — original tokens had them stripped
            for t in score.tracks:
                t.attributes.clear()
            # Create encoder with multi-fill config
            mf_cfg = _core.EncoderConfig.from_json(cfg.to_json())
            mf_cfg.supports_infill = True
            mf_vocab = _core.Vocabulary(mf_cfg)
            mf_enc = _core.Encoder(mf_vocab)
            mf_opts = _core.EncodeOptions()
            mf_opts.multi_fill = fill_set
            ref = mf_enc.encode(score, mf_opts)

            mapping_mf = build_orig_to_ref_mapping(yellow_orig_encoder, mf_vocab)
            orig_remapped_mf = remap_orig_tokens(orig, mapping_mf)

            compared += 1
            if orig_remapped_mf != ref:
                report = diff_report(
                    "orig_remapped_mf", orig_remapped_mf, lambda t: pretty_ref(mf_vocab, t),
                    "ref",              ref,              lambda t: pretty_ref(mf_vocab, t),
                )
                failures.append((mp.split('/')[-1], report))

        if compared == 0:
            pytest.skip("no comparable files")
        if failures:
            msg = f"\nMulti-fill parity failed on {len(failures)}/{compared}:\n"
            for name, report in failures[:3]:
                msg += f"\n--- {name} ---\n{report}\n"
            pytest.fail(msg)


# ---------------------------------------------------------------------------
# StepPlanner parity
# ---------------------------------------------------------------------------

class TestStepPlanner:
    """Verify StepPlanner produces sensible steps for various configurations."""

    def test_ar_single_track_4bars(self, yellow_ref_components):
        """1 track, 4 bars, all selected, AR → should produce steps covering all bars."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components
        mask = _core.SelectionMask()
        mask.selected = [[True, True, True, True]]
        mask.autoregressive = [True]
        mask.ignore = [False]

        planner = _core.StepPlanner(mask, cfg, bars_per_step=1, tracks_per_step=1)
        steps = planner.plan()
        assert len(steps) > 0, "should produce at least 1 step"

        # All 4 bars should be covered
        generated = set()
        for s in steps:
            assert s.is_autoregressive
            for t, b in s.bars_to_generate:
                generated.add((t, b))
        assert generated == {(0, 0), (0, 1), (0, 2), (0, 3)}, \
            f"expected all 4 bars generated, got {generated}"

    def test_infill_single_track(self, yellow_ref_components):
        """1 track, 4 bars, bars 1-2 selected, infill → should produce infill steps."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components
        mask = _core.SelectionMask()
        mask.selected = [[False, True, True, False]]
        mask.autoregressive = [False]
        mask.ignore = [False]

        planner = _core.StepPlanner(mask, cfg, bars_per_step=1, tracks_per_step=1)
        steps = planner.plan()
        assert len(steps) > 0

        generated = set()
        for s in steps:
            assert not s.is_autoregressive
            for t, b in s.bars_to_generate:
                generated.add((t, b))
        assert generated == {(0, 1), (0, 2)}, f"got {generated}"

    def test_infill_step_has_context(self, yellow_ref_components):
        """Infill steps should have context bars (surrounding bars)."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components
        mask = _core.SelectionMask()
        mask.selected = [[False, True, False, False]]
        mask.autoregressive = [False]
        mask.ignore = [False]

        planner = _core.StepPlanner(mask, cfg, bars_per_step=1, tracks_per_step=1)
        steps = planner.plan()
        assert len(steps) == 1
        s = steps[0]

        # Context should include non-selected bars in the window
        has_context = False
        for ti in range(len(s.context)):
            for bi in range(len(s.context[ti])):
                if s.context[ti][bi]:
                    has_context = True
        assert has_context, "infill step should have context bars"

    def test_bar_mapping_populated(self, yellow_ref_components):
        """bar_mapping should be populated for all bars_to_generate."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components
        mask = _core.SelectionMask()
        mask.selected = [[True, True, True, True]]
        mask.autoregressive = [True]
        mask.ignore = [False]

        planner = _core.StepPlanner(mask, cfg, bars_per_step=1, tracks_per_step=1)
        steps = planner.plan()

        for s in steps:
            n_gen = len(s.bars_to_generate)
            n_map = len(s.bar_mapping)
            assert n_map == n_gen, (
                f"bar_mapping has {n_map} entries but bars_to_generate has {n_gen}"
            )

    def test_window_within_model_dim(self, yellow_ref_components):
        """Every step's window should be ≤ model_dim bars wide."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components
        mask = _core.SelectionMask()
        mask.selected = [[True] * 8]
        mask.autoregressive = [True]
        mask.ignore = [False]

        planner = _core.StepPlanner(mask, cfg, bars_per_step=1, tracks_per_step=1)
        steps = planner.plan()

        for s in steps:
            window = s.end_bar - s.start_bar
            assert window <= cfg.model_dim, (
                f"window {window} > model_dim {cfg.model_dim}"
            )


# ---------------------------------------------------------------------------
# Randomized fuzz tests
# ---------------------------------------------------------------------------

class TestRandomizedModes:
    """Generate many random generation tasks and verify invariants."""

    N_RANDOM_TASKS = 50
    SEED = 42

    def _random_tasks(self):
        rng = random.Random(self.SEED)
        tasks = []
        for _ in range(self.N_RANDOM_TASKS):
            n_bars = rng.choice([4, 8])
            n_tracks = rng.randint(1, 3)
            mode = rng.choice(["ar", "infill", "mixed"])

            selected = []
            autoregressive = []
            ignore = []
            for t in range(n_tracks):
                if mode == "ar":
                    ar = True
                elif mode == "infill":
                    ar = False
                else:
                    ar = rng.choice([True, False])
                autoregressive.append(ar)
                ignore.append(False)

                # Random bar selection
                track_sel = [False] * n_bars
                n_selected = rng.randint(1, n_bars)
                start = rng.randint(0, n_bars - n_selected)
                for b in range(start, start + n_selected):
                    track_sel[b] = True
                selected.append(track_sel)

            bps = rng.choice([1, 2, 4])
            tps = rng.choice([1])
            tasks.append({
                "n_bars": n_bars, "n_tracks": n_tracks,
                "selected": selected, "autoregressive": autoregressive,
                "ignore": ignore, "bars_per_step": bps, "tracks_per_step": tps,
                "mode": mode,
            })
        return tasks

    def test_step_planner_covers_all_selected_bars(self, yellow_ref_components):
        """Every selected bar must appear in exactly one step's bars_to_generate."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components

        for task in self._random_tasks():
            mask = _core.SelectionMask()
            mask.selected = task["selected"]
            mask.autoregressive = task["autoregressive"]
            mask.ignore = task["ignore"]

            planner = _core.StepPlanner(
                mask, cfg, task["bars_per_step"], task["tracks_per_step"]
            )
            steps = planner.plan()

            # Collect all generated bars
            generated = set()
            for s in steps:
                for tb in s.bars_to_generate:
                    generated.add(tb)

            # Check all selected bars are covered
            expected = set()
            for t in range(task["n_tracks"]):
                for b in range(task["n_bars"]):
                    if task["selected"][t][b] and not task["ignore"][t]:
                        expected.add((t, b))

            assert expected == generated, (
                f"task={task}\nexpected={sorted(expected)}\ngot={sorted(generated)}"
            )

    def test_step_planner_ar_vs_infill_flag(self, yellow_ref_components):
        """Steps from AR tracks should have is_autoregressive=True, infill=False."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components

        for task in self._random_tasks():
            mask = _core.SelectionMask()
            mask.selected = task["selected"]
            mask.autoregressive = task["autoregressive"]
            mask.ignore = task["ignore"]

            planner = _core.StepPlanner(
                mask, cfg, task["bars_per_step"], task["tracks_per_step"]
            )
            steps = planner.plan()

            for s in steps:
                for t, b in s.bars_to_generate:
                    if task["autoregressive"][t]:
                        assert s.is_autoregressive, (
                            f"step generating AR track {t} should be is_autoregressive=True"
                        )
                    else:
                        assert not s.is_autoregressive, (
                            f"step generating infill track {t} should be is_autoregressive=False"
                        )

    def test_step_windows_valid(self, yellow_ref_components):
        """All step windows should be valid: 0 ≤ start < end, end-start ≤ model_dim."""
        import midigpt_refactor._core as _core
        cfg, vocab, _, _ = yellow_ref_components

        for task in self._random_tasks():
            mask = _core.SelectionMask()
            mask.selected = task["selected"]
            mask.autoregressive = task["autoregressive"]
            mask.ignore = task["ignore"]

            planner = _core.StepPlanner(
                mask, cfg, task["bars_per_step"], task["tracks_per_step"]
            )
            steps = planner.plan()

            for s in steps:
                assert 0 <= s.start_bar < s.end_bar, (
                    f"invalid window [{s.start_bar}, {s.end_bar})"
                )
                assert s.end_bar - s.start_bar <= cfg.model_dim, (
                    f"window too large: {s.end_bar - s.start_bar} > {cfg.model_dim}"
                )

    def test_encoding_roundtrip_random(self, yellow_ref_components):
        """Random AR/infill encoding should roundtrip through encode → decode."""
        import midigpt_refactor._core as _core
        cfg, vocab, ref_enc, ref_dec = yellow_ref_components
        midis = _get_test_midis(4, max_files=1000)
        if not midis:
            pytest.skip("no test MIDI available")

        rng = random.Random(self.SEED)
        for mp in midis:
            # Test multi-fill roundtrip with random bars
            n_fill = rng.randint(1, 3)
            fill_bars = set()
            for _ in range(n_fill):
                fill_bars.add((0, rng.randint(0, 3)))

            ref = _ref_encode(vocab, ref_enc, mp, 4,
                              do_multi_fill=True, multi_fill=fill_bars)
            decoded = ref_dec.decode(ref)
            assert len(decoded.tracks) > 0, "decoded should have tracks"
            assert len(decoded.tracks[0].bars) == 4, \
                f"decoded should have 4 bars, got {len(decoded.tracks[0].bars)}"
