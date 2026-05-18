"""Decode parity: feed the SAME token sequence to both decoders and verify
they produce the same notes (pitch, velocity, onset_ticks, duration_ticks)
per (track, bar).

Strategy:
  1. Encode a trimmed MIDI with the ORIGINAL Yellow encoder (known-good).
  2. Decode the resulting tokens with both codebases.
  3. Extract a normalized note set keyed by (track_idx, bar_idx) from each.
  4. Compare.

Notes:
  - Original event format = note_on / note_off pairs (velocity 0 => off);
    we pair them within each bar to recover (pitch, vel, onset, duration).
  - Refactored format = explicit Note(pitch, vel, onset_ticks, duration_ticks),
    indexed from Bar.note_indices into Score.notes.
"""
import json
import pytest
from .conftest import silence_stdio, midi_files


def _orig_notes_by_bar(piece_json: str, ref_score=None) -> dict:
    """{(track_idx, bar_idx): sorted [(pitch, vel, onset_ticks, dur)]} from
    original tokens_to_json output. Pairs note_on/note_off across bars within
    a track — the original encoder splits cross-bar notes (note_on in bar N,
    note_off in bar N+1). We attribute the resulting note to the bar where it
    starts and use absolute ticks for onset to match the refactored decoder."""
    d = json.loads(piece_json)
    events = d.get("events", [])
    resolution = d.get("resolution", 12)
    out = {}
    for ti, t in enumerate(d.get("tracks", [])):
        bars = t.get("bars", [])
        out_per_bar = {bi: [] for bi in range(len(bars))}
        opens = {}  # pitch -> (abs_on_tick, vel, on_bar_idx)
        bar_start_tick = 0
        # Resolve bar tick lengths from refactored score when orig JSON has no ts
        ref_bar_ticks = []
        if ref_score is not None and ti < len(ref_score.tracks):
            for rb in ref_score.tracks[ti].bars:
                bl = rb.beat_length if rb.beat_length > 0 else 4
                ref_bar_ticks.append(bl * resolution)
        for bi, b in enumerate(bars):
            ts_num = b.get("ts_numerator") or b.get("tsNumerator")
            ts_den = b.get("ts_denominator") or b.get("tsDenominator")
            ibl = b.get("internal_beat_length") or b.get("internalBeatLength")
            if ts_num and ts_den and (4 * ts_num // ts_den) > 0:
                bar_ticks = (4 * ts_num // ts_den) * resolution
            elif ibl and ibl > 0:
                bar_ticks = ibl * resolution
            elif bi < len(ref_bar_ticks) and ref_bar_ticks[bi] > 0:
                bar_ticks = ref_bar_ticks[bi]
            else:
                bar_ticks = 4 * resolution
            for ei in b.get("events", []):
                ev = events[ei]
                p, v, time = ev["pitch"], ev["velocity"], ev["time"]
                abs_time = bar_start_tick + time
                if v > 0:
                    opens[p] = (abs_time, time, v, bi)
                else:
                    if p in opens:
                        on_abs, on_local, vel, on_bar = opens.pop(p)
                        out_per_bar[on_bar].append((p, vel, on_local, abs_time - on_abs))
            bar_start_tick += bar_ticks
        for bi in range(len(bars)):
            out[(ti, bi)] = sorted(out_per_bar[bi])
    return out


def _ref_notes_by_bar(score) -> dict:
    """{(track_idx, bar_idx): sorted [(pitch, vel, onset_ticks, duration_ticks)]}
    from refactored Score."""
    out = {}
    for ti, t in enumerate(score.tracks):
        for bi, b in enumerate(t.bars):
            notes = []
            for ni in b.note_indices:
                n = score.notes[ni]
                notes.append((n.pitch, n.velocity, n.onset_ticks, n.duration_ticks))
            out[(ti, bi)] = sorted(notes)
    return out


def _format_diff(orig, ref, max_lines=20):
    keys = sorted(set(orig) | set(ref))
    lines = []
    for k in keys:
        a = orig.get(k, [])
        b = ref.get(k, [])
        if a != b:
            lines.append(f"  (track={k[0]}, bar={k[1]}):")
            lines.append(f"    orig({len(a):>3}): {a[:6]}{' ...' if len(a)>6 else ''}")
            lines.append(f"    ref ({len(b):>3}): {b[:6]}{' ...' if len(b)>6 else ''}")
            if len(lines) > max_lines:
                lines.append("    ...")
                break
    return "\n".join(lines)


def _bench_files():
    """Files where the original encoder accepts a 4-bar trim."""
    return midi_files()


class TestDecodeParity:
    def test_same_tokens_same_notes(self, yellow_orig_encoder, yellow_ref_components):
        cfg, vocab, ref_enc, ref_dec = yellow_ref_components

        compared = 0
        mismatches = []
        for mp in _bench_files():
            with silence_stdio():
                try:
                    piece_json = yellow_orig_encoder.midi_to_json(str(mp))
                except Exception:
                    continue
            d = json.loads(piece_json)
            for t in d.get("tracks", []):
                t["bars"] = t.get("bars", [])[:4]
            piece_json_4 = json.dumps(d)

            with silence_stdio():
                try:
                    tokens = yellow_orig_encoder.json_to_tokens(piece_json_4)
                except Exception:
                    continue
                # Decode with both
                orig_decoded_json = yellow_orig_encoder.tokens_to_json(tokens)
                ref_score = ref_dec.decode(tokens)

            orig_notes = _orig_notes_by_bar(orig_decoded_json, ref_score)
            ref_notes = _ref_notes_by_bar(ref_score)

            # Restrict comparison to track/bar keys both produced
            common = sorted(set(orig_notes) & set(ref_notes))
            if not common:
                continue

            ok = all(orig_notes[k] == ref_notes[k] for k in common)
            compared += 1
            if not ok:
                mismatches.append((mp.name, _format_diff(orig_notes, ref_notes)))

        if compared == 0:
            pytest.skip("no comparable MIDI files — original encoder rejected all")

        if mismatches:
            msg = f"\nDecode parity failed on {len(mismatches)}/{compared} files:\n"
            for name, diff in mismatches[:3]:
                msg += f"\n--- {name} ---\n{diff}\n"
            pytest.fail(msg)
