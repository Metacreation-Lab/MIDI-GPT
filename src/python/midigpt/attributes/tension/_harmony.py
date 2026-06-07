# Ported from tension_calculation.py (symusic version).
# Based on R. Guo's midi-miner code, modified to remove pretty_midi and use symusic.
# CLI helpers (extract_notes, walk, get_args) and argparse are excluded.

import copy
import itertools
import os
import warnings
from typing import List, Tuple

import numpy as np
from numpy import ndarray

import symusic

PianoRoll = ndarray

major_enharmonics = {"C#": "D-", "D#": "E-", "F#": "G-", "G#": "A-", "A#": "B-"}
minor_enharmonics = {"D-": "C#", "D#": "E-", "G-": "F#", "A-": "G#", "A#": "B-"}

octave = 12

pitch_index_to_sharp_names = np.array(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
)
pitch_index_to_flat_names = np.array(
    ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
)

pitch_name_to_pitch_index = {
    "G-": -6, "D-": -5, "A-": -4, "E-": -3,
    "B-": -2, "F": -1, "C": 0, "G": 1, "D": 2, "A": 3,
    "E": 4, "B": 5, "F#": 6, "C#": 7, "G#": 8, "D#": 9, "A#": 10,
}
pitch_index_to_pitch_name = {v: k for k, v in pitch_name_to_pitch_index.items()}

valid_major = ["G-", "D-", "A-", "E-", "B-", "F", "C", "G", "D", "A", "E", "B"]
valid_minor = ["E-", "B-", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#"]

enharmonic_dict = {"F#": "G-", "C#": "D-", "G#": "A-", "D#": "E-", "A#": "B-"}
enharmonic_reverse_dict = {v: k for k, v in enharmonic_dict.items()}

all_key_names = [
    "C major", "G major", "D major", "A major",
    "E major", "B major", "F major", "B- major",
    "E- major", "A- major", "D- major", "G- major",
    "A minor", "E minor", "B minor", "F# minor",
    "C# minor", "G# minor", "D minor", "G minor",
    "C minor", "F minor", "B- minor", "E- minor",
]

note_index_to_pitch_index = [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5]

weight = np.array([0.536, 0.274, 0.19])
alpha = 0.75
beta = 0.75
verticalStep = 0.4
radius = 1.0


# ---------------------------------------------------------------------------
# symusic helpers
# ---------------------------------------------------------------------------

def _get_tpq(score) -> int:
    for attr in ("tpq", "ticks_per_quarter", "resolution"):
        if hasattr(score, attr):
            return int(getattr(score, attr))
    raise AttributeError("Could not find ticks-per-quarter on symusic.Score")


def _iter_notes(score):
    for tr in getattr(score, "tracks", []):
        for n in getattr(tr, "notes", []):
            yield n


def _note_start_tick(n) -> int:
    for attr in ("start", "start_tick", "time", "onset"):
        if hasattr(n, attr):
            return int(getattr(n, attr))
    raise AttributeError("Could not find note start tick")


def _note_end_tick(n) -> int:
    for attr in ("end", "end_tick", "endTime", "offset"):
        if hasattr(n, attr):
            return int(getattr(n, attr))
    if hasattr(n, "duration"):
        return int(_note_start_tick(n) + getattr(n, "duration"))
    raise AttributeError("Could not find note end tick")


def _note_pitch(n) -> int:
    for attr in ("pitch", "key", "midi"):
        if hasattr(n, attr):
            return int(getattr(n, attr))
    raise AttributeError("Could not find note pitch")


def _is_drum_track(track) -> bool:
    if hasattr(track, "is_drum"):
        return bool(track.is_drum)
    if hasattr(track, "drum"):
        return bool(track.drum)
    if hasattr(track, "channel"):
        return int(track.channel) == 9
    return False


def _get_end_tick(score) -> int:
    end_tick = 0
    for n in _iter_notes(score):
        end_tick = max(end_tick, _note_end_tick(n))
    for lst_name in ("tempos", "time_signatures"):
        for ev in getattr(score, lst_name, []):
            if hasattr(ev, "time"):
                end_tick = max(end_tick, int(ev.time))
    return int(end_tick)


def _get_tempo_map(score):
    tpq = _get_tpq(score)
    tempos = list(getattr(score, "tempos", []))
    if not tempos:
        class _T:
            time = 0
            qpm = 120.0
        tempos = [_T()]

    tempos = sorted(tempos, key=lambda t: int(t.time))
    ticks = np.array([int(t.time) for t in tempos], dtype=np.int64)
    qpm = np.array([float(getattr(t, "qpm", 120.0)) for t in tempos], dtype=np.float64)

    secs = np.zeros_like(qpm)
    for i in range(1, len(ticks)):
        dticks = ticks[i] - ticks[i - 1]
        sec_per_tick = 60.0 / (qpm[i - 1] * tpq)
        secs[i] = secs[i - 1] + dticks * sec_per_tick

    return ticks, secs, qpm


def tick_to_time(score, tick_values: ndarray) -> ndarray:
    tick_values = np.asarray(tick_values, dtype=np.int64)
    tempo_ticks, tempo_secs, tempo_qpm = _get_tempo_map(score)
    tpq = _get_tpq(score)

    idx = np.searchsorted(tempo_ticks, tick_values, side="right") - 1
    idx = np.clip(idx, 0, len(tempo_ticks) - 1)

    base_tick = tempo_ticks[idx]
    base_sec = tempo_secs[idx]
    sec_per_tick = 60.0 / (tempo_qpm[idx] * tpq)

    return base_sec + (tick_values - base_tick) * sec_per_tick


def time_to_tick(score, sec_values: ndarray) -> ndarray:
    sec_values = np.asarray(sec_values, dtype=np.float64)
    tempo_ticks, tempo_secs, tempo_qpm = _get_tempo_map(score)
    tpq = _get_tpq(score)

    idx = np.searchsorted(tempo_secs, sec_values, side="right") - 1
    idx = np.clip(idx, 0, len(tempo_secs) - 1)

    base_tick = tempo_ticks[idx]
    base_sec = tempo_secs[idx]
    ticks_per_sec = (tempo_qpm[idx] * tpq) / 60.0

    return (base_tick + (sec_values - base_sec) * ticks_per_sec).astype(np.int64)


def _get_time_signatures(score):
    tss = list(getattr(score, "time_signatures", []))
    if not tss:
        return [(0, 4, 4)]
    out = []
    for ts in tss:
        t = int(getattr(ts, "time", 0))
        num = int(getattr(ts, "numerator", 4))
        den = int(getattr(ts, "denominator", 4))
        out.append((t, num, den))
    out.sort(key=lambda x: x[0])
    if out[0][0] != 0:
        out.insert(0, (0, 4, 4))
    return out


def get_piano_roll(score, time_grid_sec: ndarray) -> PianoRoll:
    time_grid_sec = np.asarray(time_grid_sec, dtype=np.float64)
    roll = np.zeros((128, time_grid_sec.shape[0]), dtype=np.int8)

    for n in _iter_notes(score):
        p = _note_pitch(n)
        if p < 0 or p > 127:
            continue
        s_tick = _note_start_tick(n)
        e_tick = _note_end_tick(n)
        if e_tick <= s_tick:
            continue
        s_sec = float(tick_to_time(score, np.array([s_tick]))[0])
        e_sec = float(tick_to_time(score, np.array([e_tick]))[0])
        i0 = int(np.searchsorted(time_grid_sec, s_sec, side="left"))
        i1 = int(np.searchsorted(time_grid_sec, e_sec, side="left"))
        i0 = max(i0, 0)
        i1 = min(i1, time_grid_sec.shape[0])
        if i1 > i0:
            roll[p, i0:i1] = 1

    return roll


def get_beat_time(score, beat_division=4):
    tpq = _get_tpq(score)
    end_tick = _get_end_tick(score)
    if end_tick <= 0:
        return None

    ts_list = _get_time_signatures(score)
    beats_ticks = []
    downbeats_ticks = []

    for i, (t0, num, den) in enumerate(ts_list):
        t1 = ts_list[i + 1][0] if (i + 1) < len(ts_list) else end_tick
        beat_ticks = int(round(tpq * 4.0 / den))
        beat_ticks = max(1, beat_ticks)
        bar_ticks = beat_ticks * max(1, num)

        beats_ticks.extend(list(range(t0, max(t0, t1) + 1, beat_ticks)))
        downbeats_ticks.extend(list(range(t0, max(t0, t1) + 1, bar_ticks)))

    beats_ticks = np.unique(np.array(beats_ticks, dtype=np.int64))
    downbeats_ticks = np.unique(np.array(downbeats_ticks, dtype=np.int64))

    beats = tick_to_time(score, beats_ticks)
    down_beats = tick_to_time(score, downbeats_ticks)

    if down_beats.shape[0] >= 2 and beats[-1] > down_beats[-1] + 1e-9:
        down_beats = np.append(down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1])

    divided_beats = []
    for i in range(len(beats) - 1):
        dt = beats[i + 1] - beats[i]
        for j in range(beat_division):
            divided_beats.append(beats[i] + (dt / beat_division) * j)
    divided_beats.append(beats[-1])

    divided_beats = np.array(divided_beats, dtype=np.float64)
    beat_indices = [int(np.argmin(np.abs(divided_beats - b))) for b in beats]
    down_beat_indices = [int(np.argmin(np.abs(divided_beats - d))) for d in down_beats]

    return divided_beats, beats, down_beats, beat_indices, down_beat_indices


# ---------------------------------------------------------------------------
# Spiral-array core
# ---------------------------------------------------------------------------

def cal_diameter(piano_roll, key_index, key_change_beat=-1, changed_key_index=-1):
    diameters = []
    for i in range(piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:, i]):
            if j > 0:
                if i / 4 > key_change_beat and key_change_beat != -1:
                    shifted_index = index % 12 - changed_key_index
                else:
                    shifted_index = index % 12 - key_index
                if shifted_index < 0:
                    shifted_index += 12
                indices.append(note_index_to_pitch_index[shifted_index])
        diameters.append(largest_distance(indices))
    return diameters


def largest_distance(pitches):
    if len(pitches) < 2:
        return 0
    diameter = 0
    for pitch_pair in itertools.combinations(pitches, 2):
        distance = np.linalg.norm(
            pitch_index_to_position(pitch_pair[0]) - pitch_index_to_position(pitch_pair[1])
        )
        if distance > diameter:
            diameter = distance
    return diameter


def piano_roll_to_ce(piano_roll, shift):
    pitch_index = []
    for i in range(piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:, i]):
            if j > 0:
                shifted_index = index % 12 - shift
                if shifted_index < 0:
                    shifted_index += 12
                indices.append(note_index_to_pitch_index[shifted_index])
        pitch_index.append(indices)
    return ce_sum(pitch_index)


def notes_to_ce(notes, shift):
    indices = []
    for index, j in enumerate(notes):
        if j > 0:
            shifted_index = index % 12 - shift
            if shifted_index < 0:
                shifted_index += 12
            indices.append(note_index_to_pitch_index[shifted_index])
    total = np.zeros(3)
    count = 0
    for index in indices:
        total += pitch_index_to_position(index)
        count += 1
    if count != 0:
        total /= count
    return total


def pitch_index_to_position(pitch_index):
    c = pitch_index - (4 * (pitch_index // 4))
    pos = np.array([0.0, 0.0, 0.0])
    if c == 0:
        pos[1] = radius
    if c == 1:
        pos[0] = radius
    if c == 2:
        pos[1] = -radius
    if c == 3:
        pos[0] = -radius
    pos[2] = pitch_index * verticalStep
    return pos


def ce_sum(indices, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(indices)
    indices = indices[start:end]
    total = np.zeros(3)
    count = 0
    for data in indices:
        for pitch in data:
            total += pitch_index_to_position(pitch)
            count += 1
    return total / count


def major_triad_position(root_index):
    root_pos = pitch_index_to_position(root_index)
    fifth_pos = pitch_index_to_position(root_index + 1)
    third_pos = pitch_index_to_position(root_index + 4)
    return weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos


def minor_triad_position(root_index):
    root_pos = pitch_index_to_position(root_index)
    fifth_pos = pitch_index_to_position(root_index + 1)
    third_pos = pitch_index_to_position(root_index - 3)
    return weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos


def major_key_position(key_index):
    return (
        weight[0] * major_triad_position(key_index)
        + weight[1] * major_triad_position(key_index + 1)
        + weight[2] * major_triad_position(key_index - 1)
    )


def minor_key_position(key_index):
    return (
        weight[0] * minor_triad_position(key_index)
        + weight[1] * (alpha * major_triad_position(key_index + 1)
                       + (1 - alpha) * minor_triad_position(key_index + 1))
        + weight[2] * (beta * minor_triad_position(key_index - 1)
                       + (1 - beta) * major_triad_position(key_index - 1))
    )


def cal_key(piano_roll, key_names, end_ratio=0.5):
    end = int(piano_roll.shape[1] * end_ratio)
    distances = []
    key_positions = []
    key_shifts = []

    for name in key_names:
        key = name.split()[0].upper()
        mode = name.split()[1]

        if mode == "minor":
            if key not in valid_minor:
                key = enharmonic_dict.get(key, enharmonic_reverse_dict.get(key, key))
            if key not in valid_minor:
                return None
        else:
            if key not in valid_major:
                key = enharmonic_dict.get(key, enharmonic_reverse_dict.get(key, key))
            if key not in valid_major:
                return None

        key_index = pitch_name_to_pitch_index[key]
        if mode == "minor":
            key_pos = minor_key_position(3)
        else:
            key_pos = major_key_position(0)
        key_positions.append(key_pos)

        if mode == "minor":
            key_index -= 3
        key_shift_name = pitch_index_to_pitch_name[key_index]

        if key_shift_name in pitch_index_to_sharp_names:
            key_shift_for_ce = int(np.argwhere(pitch_index_to_sharp_names == key_shift_name)[0][0])
        else:
            key_shift_for_ce = int(np.argwhere(pitch_index_to_flat_names == key_shift_name)[0][0])

        key_shifts.append(key_shift_for_ce)
        ce = piano_roll_to_ce(piano_roll[:, :end], key_shift_for_ce)
        distances.append(np.linalg.norm(ce - key_pos))

    index = int(np.argmin(np.array(distances)))
    return key_names[index], key_positions[index], key_shifts[index]


def merge_tension(metric, beat_indices, down_beat_indices, window_size=-1):
    if window_size == -1:
        new_metric = [
            np.mean(metric[down_beat_indices[i]:down_beat_indices[i + 1]], axis=0)
            for i in range(len(down_beat_indices) - 1)
        ]
    else:
        new_metric = [
            np.mean(metric[beat_indices[i]:beat_indices[i + window_size]], axis=0)
            for i in range(0, len(beat_indices) - window_size, window_size)
        ]
    return np.array(new_metric)


def cal_centroid(piano_roll, key_index, key_change_beat=-1, changed_key_index=-1):
    centroids = []
    for time_step in range(piano_roll.shape[1]):
        roll = piano_roll[:, time_step]
        if key_change_beat != -1 and time_step / 4 > key_change_beat:
            centroids.append(notes_to_ce(roll, changed_key_index))
        else:
            centroids.append(notes_to_ce(roll, key_index))
    return centroids


def detect_key_change(key_diff, diameter, start_ratio=0.5):
    key_diff_ratios = []
    fill_one = False
    steps = 0

    for i in range(8, key_diff.shape[0] - 8):
        if fill_one and steps > 0:
            key_diff_ratios.append(1)
            steps -= 1
            if steps == 0:
                fill_one = False
            continue

        if np.any(key_diff[i - 4:i]) and np.any(key_diff[i:i + 4]):
            previous = np.mean(key_diff[i - 4:i])
            current = np.mean(key_diff[i:i + 4])
            key_diff_ratios.append(current / previous)
        else:
            fill_one = True
            steps = 4

    key_diff_change_bar = -1
    for i in range(int(len(key_diff_ratios) * start_ratio), len(key_diff_ratios) - 2):
        if np.mean(key_diff_ratios[i:i + 4]) > 2:
            key_diff_change_bar = i
            break

    return key_diff_change_bar + 12 if key_diff_change_bar != -1 else key_diff_change_bar


def get_key_index_change(score, start_time, sixteenth_time_sec):
    new_score = copy.deepcopy(score)
    cut_tick = int(time_to_tick(new_score, np.array([start_time]))[0])

    for tr in new_score.tracks:
        kept = [n for n in list(tr.notes) if _note_start_tick(n) >= cut_tick]
        tr.notes = kept

    piano_roll = get_piano_roll(new_score, sixteenth_time_sec)
    return cal_key(piano_roll, all_key_names, end_ratio=1)


def cal_tension(
    file_name, piano_roll, sixteenth_time, beat_time, beat_indices,
    down_beat_time, down_beat_indices, score, input_folder, output_folder,
    window_size=1, key_name="", end_ratio=0.2, key_changed=True,
):
    try:
        base_name = os.path.basename(file_name)

        key_name, key_pos, note_shift = cal_key(piano_roll, key_name, end_ratio=end_ratio)

        centroids = cal_centroid(piano_roll, note_shift, -1, -1)

        changed_note_shift = -1
        changed_key_name = ""
        key_change_beat = -1
        change_time = -1
        key_change_bar = -1

        if key_changed is True:
            merged_centroids = merge_tension(centroids, beat_indices, down_beat_indices, window_size=-1)
            merged_centroids = np.array(merged_centroids)
            silent = np.where(np.linalg.norm(merged_centroids, axis=-1) == 0)

            key_diff = merged_centroids - key_pos
            key_diff = np.linalg.norm(key_diff, axis=-1)
            key_diff[silent] = 0

            diameters = cal_diameter(piano_roll, note_shift, -1, -1)
            diameters = merge_tension(diameters, beat_indices, down_beat_indices, window_size=-1)

            key_change_bar = detect_key_change(key_diff, diameters, start_ratio=end_ratio)

            if key_change_bar != -1:
                change_time = down_beat_time[key_change_bar]
                result = get_key_index_change(score, change_time, sixteenth_time)
                if result is not None:
                    changed_key_name, changed_key_pos, changed_note_shift = result
                    if changed_key_name == key_name:
                        changed_note_shift = -1
                        changed_key_name = ""
                        key_change_beat = -1
                        change_time = -1
                        key_change_bar = -1

        centroids = cal_centroid(piano_roll, note_shift, -1, changed_note_shift)
        merged_centroids = merge_tension(centroids, beat_indices, down_beat_indices, window_size=window_size)
        merged_centroids = np.array(merged_centroids)
        silent = np.where(np.linalg.norm(merged_centroids, axis=-1) < 0.1)

        if window_size == -1:
            window_time = down_beat_time
        else:
            window_time = beat_time[::window_size]

        key_diff = np.linalg.norm(merged_centroids - key_pos, axis=-1)
        key_diff[silent] = 0

        diameters = cal_diameter(piano_roll, note_shift, -1, changed_note_shift)
        diameters = merge_tension(diameters, beat_indices, down_beat_indices, window_size)
        diameters[silent] = 0

        centroid_diff = np.diff(merged_centroids, axis=0)
        np.nan_to_num(centroid_diff, copy=False)
        centroid_diff = np.linalg.norm(centroid_diff, axis=-1)
        centroid_diff = np.insert(centroid_diff, 0, 0)

        total_tension = key_diff

        if input_folder and input_folder[-1] != "/":
            input_folder += "/"
        name_with_sub_folder = file_name.replace(input_folder, "")
        output_name = os.path.join(output_folder, name_with_sub_folder)
        new_output_folder = os.path.dirname(output_name)

        times = window_time[: len(total_tension)]
        return [
            total_tension, diameters, centroid_diff,
            key_name, change_time, key_change_bar, changed_key_name,
            new_output_folder, times,
        ]

    except Exception as e:
        warnings.warn(f"Unexpected error computing tension for {file_name}: {e}")
        return None
