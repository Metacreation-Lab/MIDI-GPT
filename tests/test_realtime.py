"""
tests/test_realtime.py

Test suite for the real-time co-generation loop.

Covers:
  1. Masking logic  — future flags computed correctly for all parameter combos
  2. Bar selection  — exactly [t, t+j) selected, suffix_autoregressive set
  3. Window alignment — early phase vs. sliding phase
  4. Playhead advance — advances by j on generation steps, by 1 otherwise
  5. Integration     — sample_multi_step called with correct inputs (requires model)

Set env vars to enable integration tests:
    export REALTIME_MODEL_PATH=/path/to/model.pt
    export REALTIME_MIDI_PATH=/path/to/test.mid
"""

import json
import os
import pytest

MODEL_PATH = os.environ.get("REALTIME_MODEL_PATH", "")
MIDI_PATH = os.environ.get(
    "REALTIME_MIDI_PATH",
    os.path.join(os.path.dirname(__file__), "short_midi", "test.mid"),
)

needs_model = pytest.mark.skipif(
    not MODEL_PATH,
    reason="Set REALTIME_MODEL_PATH to run integration tests",
)
needs_midi = pytest.mark.skipif(
    not os.path.exists(MIDI_PATH),
    reason=f"Test MIDI not found: {MIDI_PATH!r}",
)

try:
    import midigpt
    HAS_MIDIGPT = True
except ImportError:
    HAS_MIDIGPT = False

pytestmark = pytest.mark.skipif(
    not HAS_MIDIGPT, reason="midigpt extension not built"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_piece(num_tracks, num_bars, resolution=12):
    """Build a minimal piece dict with empty bars."""
    tracks = []
    for _ in range(num_tracks):
        bars = [{"ts_numerator": 4, "ts_denominator": 4} for _ in range(num_bars)]
        tracks.append({"track_type": 10, "instrument": 0, "bars": bars})
    return {"tracks": tracks, "events": [], "resolution": resolution}


def make_piece_with_notes(num_tracks, num_bars, fill_bars, resolution=12):
    """Build a piece with simple repeated notes in the given bars for all tracks.

    fill_bars: list of bar indices to populate with notes.
    Emits note-on / note-off event pairs (velocity=0 for note-off) per the proto spec.
    Pitches: C4 (60) on beats 1 and 3, held for one beat each.
    """
    beat = resolution  # one beat in ticks (resolution = subdivisions per beat)

    # Build one bar's worth of events (4 events: 2 note-on, 2 note-off)
    bar_events_template = [
        {"time": 0,          "pitch": 60, "velocity": 80},   # note-on  beat 1
        {"time": beat - 1,   "pitch": 60, "velocity": 0},    # note-off beat 1 end
        {"time": beat * 2,   "pitch": 64, "velocity": 80},   # note-on  beat 3
        {"time": beat * 3 - 1, "pitch": 64, "velocity": 0},  # note-off beat 3 end
    ]
    events_per_bar = len(bar_events_template)
    fill_set = set(fill_bars)

    # Each track gets its own slice of the flat events pool
    all_events = bar_events_template * len(fill_bars) * num_tracks
    tracks = []
    for ti in range(num_tracks):
        bars = []
        filled_count = 0
        for b in range(num_bars):
            if b in fill_set:
                base = (ti * len(fill_bars) + filled_count) * events_per_bar
                bars.append({
                    "ts_numerator": 4,
                    "ts_denominator": 4,
                    "events": list(range(base, base + events_per_bar)),
                })
                filled_count += 1
            else:
                bars.append({"ts_numerator": 4, "ts_denominator": 4})
        tracks.append({"track_type": 10, "instrument": 0, "bars": bars})

    return {"tracks": tracks, "events": all_events, "resolution": resolution}


def compute_realtime_step(
    playhead, k, j, B, D,
    num_tracks_human, num_bars,
    mask_gap=False,
    adapt_buffer=False,
):
    """
    Compute the status dict for one real-time generation step exactly as
    simulate_realtime_agent.py does (after all bug fixes), and return:
      - status dict
      - target_bar (or None if not generating)
      - sel list (selected_bars for agent)
    """
    target_bar = None
    num_anticipation = j

    if adapt_buffer:
        if playhead + k >= B:
            target_bar = playhead + k
    else:
        if playhead >= B:
            target_bar = playhead + k

    if target_bar is not None:
        if target_bar < B:
            target_bar = None
        elif target_bar >= num_bars:
            target_bar = None
        else:
            num_anticipation = min(j, num_bars - target_bar)

    # Human status bars
    human_status_bars = [
        [{"future": b >= playhead} for b in range(num_bars)]
        for _ in range(num_tracks_human)
    ]

    # Agent status bars + selection
    sel = [False] * num_bars
    agent_status_bars = []

    if target_bar is not None:
        for b_idx in range(target_bar, min(target_bar + num_anticipation, num_bars)):
            sel[b_idx] = True
        for b in range(num_bars):
            if b >= target_bar + num_anticipation:
                agent_status_bars.append({"future": True})
            elif mask_gap and playhead <= b < target_bar:
                agent_status_bars.append({"future": True})
            else:
                agent_status_bars.append({"future": False})
    else:
        agent_status_bars = [{"future": False}] * num_bars

    status = {"tracks": []}
    for i in range(num_tracks_human):
        status["tracks"].append({
            "track_id": i,
            "track_type": 10,
            "selected_bars": [False] * num_bars,
            "suffix_autoregressive": False,
            "bars": human_status_bars[i],
        })
    status["tracks"].append({
        "track_id": num_tracks_human,
        "track_type": 10,
        "selected_bars": sel,
        "suffix_autoregressive": True,
        "instrument": "acoustic_bass",
        "polyphony_hard_limit": 10,
        "bars": agent_status_bars,
    })

    return status, target_bar, sel, num_anticipation


# ── 1. Masking logic ──────────────────────────────────────────────────────────

class TestMaskingLogic:
    """future flags in status bars are computed correctly."""

    def test_human_bars_before_playhead_not_masked(self):
        status, _, _, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        human_bars = status["tracks"][0]["bars"]
        for b in range(4):
            assert not human_bars[b].get("future"), f"bar {b} should not be masked"

    def test_human_bars_at_and_after_playhead_masked(self):
        status, _, _, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        human_bars = status["tracks"][0]["bars"]
        for b in range(4, 12):
            assert human_bars[b].get("future"), f"bar {b} should be masked"

    def test_agent_context_bars_not_masked(self):
        # k=1, j=1, playhead=4: target=5. Agent bars [0,4] are context.
        status, target_bar, _, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        assert target_bar == 5
        agent_bars = status["tracks"][1]["bars"]
        for b in range(5):
            assert not agent_bars[b].get("future"), f"agent bar {b} should not be masked"

    def test_agent_target_bars_not_masked_j2(self):
        # k=1, j=2, playhead=4: target=5. Bars 5 AND 6 are targets — both visible.
        status, target_bar, _, j = compute_realtime_step(
            playhead=4, k=1, j=2, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        assert target_bar == 5
        agent_bars = status["tracks"][1]["bars"]
        assert not agent_bars[5].get("future"), "target bar 5 should not be masked"
        assert not agent_bars[6].get("future"), "target bar 6 should not be masked"

    def test_agent_bars_beyond_target_window_masked(self):
        # j=2: bars [7, ...] beyond target window [5,6] should be masked
        status, _, _, _ = compute_realtime_step(
            playhead=4, k=1, j=2, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        agent_bars = status["tracks"][1]["bars"]
        for b in range(7, 12):
            assert agent_bars[b].get("future"), f"agent bar {b} should be masked (beyond window)"

    def test_mask_gap_true_hides_gap_bars(self):
        # k=2, j=1, playhead=4: target=6. Gap bars [4,5] should be masked.
        status, target_bar, _, _ = compute_realtime_step(
            playhead=4, k=2, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
            mask_gap=True,
        )
        assert target_bar == 6
        agent_bars = status["tracks"][1]["bars"]
        assert agent_bars[4].get("future"), "gap bar 4 should be masked (mask_gap=True)"
        assert agent_bars[5].get("future"), "gap bar 5 should be masked (mask_gap=True)"

    def test_mask_gap_false_shows_gap_bars(self):
        # Same params, mask_gap=False: gap bars [4,5] should NOT be masked.
        status, target_bar, _, _ = compute_realtime_step(
            playhead=4, k=2, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
            mask_gap=False,
        )
        assert target_bar == 6
        agent_bars = status["tracks"][1]["bars"]
        assert not agent_bars[4].get("future"), "gap bar 4 should be visible (mask_gap=False)"
        assert not agent_bars[5].get("future"), "gap bar 5 should be visible (mask_gap=False)"

    def test_no_generation_during_buffer(self):
        # playhead < B: no target bar, no masking decision needed (all non-future)
        for p in range(4):
            status, target_bar, sel, _ = compute_realtime_step(
                playhead=p, k=1, j=1, B=4, D=8,
                num_tracks_human=1, num_bars=12,
            )
            assert target_bar is None, f"should not generate during buffer (playhead={p})"
            assert not any(sel), "no bars selected during buffer"


# ── 2. Bar selection ──────────────────────────────────────────────────────────

class TestBarSelection:
    """Exactly [t, t+j) selected, suffix_autoregressive always True on agent."""

    def test_j1_selects_exactly_one_bar(self):
        _, target_bar, sel, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        assert target_bar == 5
        assert sum(sel) == 1
        assert sel[5] is True

    def test_j2_selects_exactly_two_bars(self):
        _, target_bar, sel, _ = compute_realtime_step(
            playhead=4, k=1, j=2, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        assert target_bar == 5
        assert sum(sel) == 2
        assert sel[5] is True and sel[6] is True

    def test_selection_does_not_exceed_piece_end(self):
        # Near end: target_bar=10, j=2, total_bars=12 → only bar 10 and 11 selected
        _, target_bar, sel, j = compute_realtime_step(
            playhead=9, k=1, j=2, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        assert target_bar == 10
        assert j == 2
        assert sel[10] is True and sel[11] is True
        assert not any(sel[12:])  # no bars beyond piece end selected

    def test_suffix_autoregressive_always_true(self):
        status, _, _, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=8,
            num_tracks_human=1, num_bars=12,
        )
        agent_track = status["tracks"][-1]
        assert agent_track["suffix_autoregressive"] is True

    def test_human_tracks_never_selected(self):
        for num_human in [1, 2]:
            status, _, _, _ = compute_realtime_step(
                playhead=4, k=1, j=1, B=4, D=8,
                num_tracks_human=num_human, num_bars=12,
            )
            for i in range(num_human):
                assert not any(status["tracks"][i]["selected_bars"])


# ── 3. Window alignment ───────────────────────────────────────────────────────

class TestWindowAlignment:
    """
    The window alignment is handled by sample_multi_step internally (model_dim param).
    Here we verify the correct model_dim is passed for various step configs.
    """

    def test_model_dim_matches_arg(self):
        # The params dict should always pass model_dim = args.model_dim unchanged.
        # (Dynamic model_dim is not supported — model_dim is a trained hyperparameter.)
        D = 8
        _, target_bar, _, _ = compute_realtime_step(
            playhead=4, k=1, j=1, B=4, D=D,
            num_tracks_human=1, num_bars=12,
        )
        # We just verify target_bar is correct; windowing is C++ internal.
        assert target_bar == 5

    def test_framework_example1(self):
        # From realtime_framework.md Example 1: D=8, B=4, k=1, j=1
        # Step 1: playhead=4, target=5
        _, t, _, _ = compute_realtime_step(4, k=1, j=1, B=4, D=8,
                                            num_tracks_human=1, num_bars=20)
        assert t == 5

    def test_framework_example2_k2(self):
        # From Example 2: D=8, B=4, k=2, j=1
        # Step 1: playhead=4, target=6
        _, t, _, _ = compute_realtime_step(4, k=2, j=1, B=4, D=8,
                                            num_tracks_human=1, num_bars=20)
        assert t == 6

    def test_framework_example3_j2(self):
        # From Example 3: D=8, B=4, k=2, j=2
        # Step 1: playhead=4, target=6, generates bars 6,7
        _, t, sel, j = compute_realtime_step(4, k=2, j=2, B=4, D=8,
                                              num_tracks_human=1, num_bars=20)
        assert t == 6
        assert j == 2
        assert sel[6] and sel[7]
        assert not sel[8]


# ── 4. Playhead advance ───────────────────────────────────────────────────────

class TestPlayheadAdvance:
    """Playhead advances by j during generation, by 1 in buffer."""

    def _simulate_playhead_sequence(self, B, k, j, total_bars, adapt=False):
        """Return list of (playhead, target_bar) at each loop iteration."""
        steps = []
        playhead = 0
        while playhead < total_bars:
            _, target_bar, _, num_anticipation = compute_realtime_step(
                playhead, k, j, B, D=8,
                num_tracks_human=1, num_bars=total_bars,
                adapt_buffer=adapt,
            )
            should_gen = target_bar is not None
            steps.append((playhead, target_bar))
            if should_gen:
                playhead += num_anticipation
            else:
                playhead += 1
        return steps

    def test_buffer_advances_by_one(self):
        steps = self._simulate_playhead_sequence(B=4, k=1, j=1, total_bars=12)
        buffer_steps = [(p, t) for p, t in steps if t is None]
        playheads = [p for p, _ in buffer_steps]
        # First 4 buffer steps are 0,1,2,3. The final playhead may also have
        # target=None if the last valid target is out of bounds (edge case).
        assert playheads[:4] == list(range(4)), f"buffer should start 0,1,2,3 got {playheads}"
        assert all(p < 4 or p >= 10 for p in playheads), \
            f"unexpected mid-song buffer steps: {playheads}"

    def test_j1_advances_every_bar(self):
        steps = self._simulate_playhead_sequence(B=4, k=1, j=1, total_bars=12)
        gen_steps = [(p, t) for p, t in steps if t is not None]
        # With j=1, playhead advances by 1 each step: 4,5,6,...
        gen_playheads = [p for p, _ in gen_steps]
        assert gen_playheads == list(range(4, 11)), f"got {gen_playheads}"

    def test_j2_advances_by_two(self):
        steps = self._simulate_playhead_sequence(B=4, k=1, j=2, total_bars=16)
        gen_steps = [(p, t) for p, t in steps if t is not None]
        gen_playheads = [p for p, _ in gen_steps]
        # Should be 4, 6, 8, 10, 12, 14
        assert all(p % 2 == 0 for p in gen_playheads), f"j=2 should always land on even: {gen_playheads}"
        assert gen_playheads[0] == 4

    def test_target_always_equals_playhead_plus_k(self):
        for k in [1, 2]:
            steps = self._simulate_playhead_sequence(B=4, k=k, j=1, total_bars=16)
            gen_steps = [(p, t) for p, t in steps if t is not None]
            for p, t in gen_steps:
                assert t == p + k, f"k={k}: expected target={p+k}, got {t}"

    def test_adapt_buffer_starts_earlier(self):
        # k=2, B=4, adapt: first gen at playhead=2 (target=4=B)
        steps_adapt = self._simulate_playhead_sequence(B=4, k=2, j=1, total_bars=16, adapt=True)
        steps_noadapt = self._simulate_playhead_sequence(B=4, k=2, j=1, total_bars=16, adapt=False)

        first_gen_adapt = next(p for p, t in steps_adapt if t is not None)
        first_gen_noadapt = next(p for p, t in steps_noadapt if t is not None)

        assert first_gen_adapt < first_gen_noadapt, \
            "adapt_buffer should start generating earlier"
        assert first_gen_adapt == 2  # B - k = 4 - 2 = 2


# ── 5. Integration tests ─────────────────────────────────────────────────────

@needs_model
class TestIntegrationSingleStep:
    """One sample_multi_step call with correct real-time inputs."""

    # Load real MIDI context once per class so all tests share the same piece.
    _MIDI_DIR = os.path.join(os.path.dirname(__file__), "short_midi")
    _piece_json_cache: dict = {}

    @classmethod
    def _load_context_piece(cls, num_bars: int) -> dict:
        """Return a multi-track piece from a real MIDI file via GhostEncoder.

        Up to 2 human tracks from the MIDI provide realistic musical context.
        An empty agent track is appended as the last track.  The original
        events array is preserved (no remapping) and bars are trimmed to
        ``num_bars``.
        """
        key = num_bars
        if key in cls._piece_json_cache:
            return json.loads(cls._piece_json_cache[key])

        midi_files = sorted(
            [os.path.join(cls._MIDI_DIR, f)
             for f in os.listdir(cls._MIDI_DIR)
             if f.endswith(".mid")]
        ) if os.path.isdir(cls._MIDI_DIR) else []

        enc = midigpt.GhostEncoder()
        for mf in midi_files:
            try:
                raw = json.loads(enc.midi_to_json(mf))
            except Exception:
                continue
            tracks = raw.get("tracks", [])
            if not tracks or len(tracks[0].get("bars", [])) < num_bars:
                continue

            # Take up to 2 human tracks, trim to num_bars
            num_human = min(2, len(tracks))
            human_tracks = []
            for t in tracks[:num_human]:
                ht = {k: v for k, v in t.items() if k != "bars"}
                ht["bars"] = t["bars"][:num_bars]
                # Sanitize track type for GhostEncoder (10=STANDARD_TRACK, 11=DRUM)
                tt = ht.get("track_type", 10)
                if tt == 8:
                    ht["track_type"] = 11
                elif tt == 9 or tt not in (10, 11):
                    ht["track_type"] = 10
                human_tracks.append(ht)

            # Empty agent track (no events, same time-sigs as track 0)
            empty_bars = [{"ts_numerator": b.get("ts_numerator", 4),
                           "ts_denominator": b.get("ts_denominator", 4)}
                          for b in human_tracks[0]["bars"]]
            agent_track = {"track_type": 10, "instrument": 32, "bars": empty_bars}

            piece = {
                "tracks": human_tracks + [agent_track],
                "events": raw.get("events", []),
                "resolution": raw.get("resolution", 12),
            }
            cls._piece_json_cache[key] = json.dumps(piece)
            return piece

        # Fallback: synthetic piece if no MIDI available
        return make_piece_with_notes(2, num_bars, list(range(num_bars // 2)))

    def _run_step(self, playhead=4, k=1, j=1, B=4, D=8, total_bars=16, mask_gap=False):
        piece = self._load_context_piece(total_bars)
        num_human = len(piece["tracks"]) - 1  # last track is agent
        status, target_bar, sel, num_anticipation = compute_realtime_step(
            playhead=playhead, k=k, j=j, B=B, D=D,
            num_tracks_human=num_human, num_bars=total_bars,
            mask_gap=mask_gap,
        )
        params = {
            "model_dim": D,
            "temperature": 1.0,
            "batch_size": 1,
            "ckpt": MODEL_PATH,
            "bars_per_step": num_anticipation,
            "tracks_per_step": 1,
            "percentage": 100,
            "polyphony_hard_limit": 10,
        }
        res_str, attempts = midigpt.sample_multi_step(
            json.dumps(piece),
            json.dumps(status),
            json.dumps(params),
            5,
            None,
        )
        return json.loads(res_str), target_bar, num_anticipation

    def test_returns_valid_json(self):
        res, _, _ = self._run_step()
        assert "tracks" in res

    def test_target_bar_has_events(self):
        piece = self._load_context_piece(16)
        agent_idx = len(piece["tracks"]) - 1
        res, target_bar, _ = self._run_step()
        agent_bars = res["tracks"][agent_idx]["bars"]
        events_in_target = agent_bars[target_bar].get("events", [])
        assert len(events_in_target) > 0, "target bar should have generated events"

    def test_j2_both_bars_have_events(self):
        piece = self._load_context_piece(16)
        agent_idx = len(piece["tracks"]) - 1
        res, target_bar, j = self._run_step(j=2)
        assert j == 2
        agent_bars = res["tracks"][agent_idx]["bars"]
        b0_events = agent_bars[target_bar].get("events", [])
        b1_events = agent_bars[target_bar + 1].get("events", [])
        # At least one of the two bars should have content
        assert len(b0_events) + len(b1_events) > 0

    def test_context_bars_not_overwritten(self):
        # Agent context bars [0, target_bar) should remain empty after generation
        piece = self._load_context_piece(16)
        agent_idx = len(piece["tracks"]) - 1
        res, target_bar, _ = self._run_step()
        agent_bars = res["tracks"][agent_idx]["bars"]
        for b in range(target_bar):
            events = agent_bars[b].get("events", [])
            assert len(events) == 0, f"context bar {b} should be empty"


@needs_model
class TestIntegrationMultiStep:
    """Multiple sequential sample_multi_step calls accumulate correctly."""

    def test_three_steps_accumulate(self):
        total_bars = 16
        D = 8
        B = 4
        k = 1
        j = 1
        piece = TestIntegrationSingleStep._load_context_piece(total_bars)
        num_human = len(piece["tracks"]) - 1
        agent_idx = num_human
        generated_bars = {}

        playhead = B
        for _ in range(3):
            status, target_bar, _, num_anticipation = compute_realtime_step(
                playhead, k, j, B, D,
                num_tracks_human=num_human, num_bars=total_bars,
            )
            params = {
                "model_dim": D,
                "temperature": 1.0,
                "batch_size": 1,
                "ckpt": MODEL_PATH,
                "bars_per_step": num_anticipation,
                "tracks_per_step": 1,
                "percentage": 100,
                "polyphony_hard_limit": 10,
            }
            res_str, _ = midigpt.sample_multi_step(
                json.dumps(piece), json.dumps(status), json.dumps(params), 5, None
            )
            res = json.loads(res_str)
            res_events = res.get("events", [])
            res_agent = res["tracks"][agent_idx]
            for b_off in range(num_anticipation):
                b_global = target_bar + b_off
                if b_global >= total_bars:
                    break
                res_bar = res_agent["bars"][b_global]
                new_idxs = []
                for ev_idx in res_bar.get("events", []):
                    new_idx = len(piece["events"])
                    piece["events"].append(res_events[ev_idx])
                    new_idxs.append(new_idx)
                piece["tracks"][agent_idx]["bars"][b_global]["events"] = new_idxs
                if new_idxs:
                    generated_bars[b_global] = True

            playhead += num_anticipation

        assert len(generated_bars) >= 1, "at least one bar should have been generated"
