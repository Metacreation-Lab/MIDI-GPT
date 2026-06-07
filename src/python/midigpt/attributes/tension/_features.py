import symusic
import numpy as np

from midigpt.attributes.tension._note import (
    NoteObj, getMelodicLine, getLoudness, getDissonance,
)
from midigpt.attributes.tension._processing import normalize
from midigpt.attributes.tension._harmony import (
    get_beat_time, get_piano_roll, cal_tension, all_key_names,
)
from midigpt.attributes.tension._model import (
    runModel,
    SAMPLE_RATE,
    MEMORY_WINDOW_DUR, MEMORY_WEIGHT,
    ATTENTIONAL_WINDOW_DUR, WINDOW_SHIFT,
    INIT_SLOPE, LAG, SLIDER_ONSET,
    FEATURE_WEIGHTS_PITCHED, FEATURE_WEIGHTS_DRUMS,
)

iOnsetFreq = 0
iMelodicContour = 1
iLoudness = 2
iTempo = 3
iHarmony = 4
iDissonance = 5
NUM_FEATURES = 6
featureList = ["Onset freq", "Melodic contour", "Loudness", "Tempo", "Harmony", "Dissonance"]

_MICRO = 1_000_000


def _get_attr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _is_drum_track(track) -> bool:
    if hasattr(track, "is_drum"):
        return bool(track.is_drum)
    if hasattr(track, "drum"):
        return bool(track.drum)
    if hasattr(track, "channel"):
        try:
            return int(track.channel) == 9
        except Exception:
            return False
    return False


def _extract_note_fields(n):
    start = _get_attr(n, "start", "time")
    end = _get_attr(n, "end")
    if end is None:
        dur = _get_attr(n, "duration")
        if dur is None:
            raise ValueError("Note has neither end nor duration")
        end = start + dur
    pitch = _get_attr(n, "pitch")
    vel = _get_attr(n, "velocity", "vel", default=64)
    if start is None or end is None or pitch is None:
        raise ValueError("Could not read note fields")
    return start, end, pitch, vel


def _build_tick_to_seconds(score):
    tpq = _get_attr(score, "tpq", "ticks_per_quarter")
    if tpq is None:
        raise ValueError("Could not find score.tpq / ticks_per_quarter")

    tempos = list(_get_attr(score, "tempos", default=[]))
    tempo_points = []
    for t in tempos:
        tick = _get_attr(t, "time", "tick", default=0)
        qpm = _get_attr(t, "qpm", "tempo", default=None)
        if qpm is None:
            continue
        tempo_points.append((int(tick), float(qpm)))

    if not tempo_points:
        tempo_points = [(0, 120.0)]

    tempo_points.sort(key=lambda x: x[0])
    if tempo_points[0][0] != 0:
        tempo_points.insert(0, (0, tempo_points[0][1]))

    start_ticks = np.array([p[0] for p in tempo_points], dtype=np.int64)
    qpm_vals = np.array([p[1] for p in tempo_points], dtype=np.float64)
    sec_per_tick = 60.0 / (qpm_vals * float(tpq))

    cum_secs = np.zeros_like(sec_per_tick)
    for i in range(1, len(start_ticks)):
        dticks = start_ticks[i] - start_ticks[i - 1]
        cum_secs[i] = cum_secs[i - 1] + dticks * sec_per_tick[i - 1]

    def tick_to_sec(tick):
        tick = np.asarray(tick, dtype=np.int64)
        idx = np.searchsorted(start_ticks, tick, side="right") - 1
        idx = np.clip(idx, 0, len(start_ticks) - 1)
        return cum_secs[idx] + (tick - start_ticks[idx]) * sec_per_tick[idx]

    return tick_to_sec


def _harmonic_tension_per_track_symusic(score, label, window_size=2, end_ratio=1.0, key_changed=False):
    beat_pack = get_beat_time(score, beat_division=4)
    if beat_pack is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = beat_pack
    piano_roll = get_piano_roll(score, sixteenth_time)

    ret = cal_tension(
        file_name=label,
        piano_roll=piano_roll,
        sixteenth_time=sixteenth_time,
        beat_time=beat_time,
        beat_indices=beat_indices,
        down_beat_time=down_beat_time,
        down_beat_indices=down_beat_indices,
        score=score,
        input_folder="",
        output_folder="",
        window_size=window_size,
        key_name=all_key_names,
        end_ratio=end_ratio,
        key_changed=key_changed,
    )

    if ret is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.asarray(ret[0], dtype=float), np.asarray(ret[-1], dtype=float)


def extractFeaturesMidi(
    inputFile,
    sampleRate=10,
    bOnsetFreq=True, bMelodicContour=True, bLoudness=True,
    bTempo=True, bHarmony=True, bDissonance=True,
    track_index=None,
    track_mode="all",
):
    score = symusic.Score(inputFile, ttype="tick")
    if hasattr(score, "sort"):
        score.sort()

    tick_to_sec = _build_tick_to_seconds(score)
    tracks = list(_get_attr(score, "tracks", default=[]))

    if not tracks:
        return np.zeros((NUM_FEATURES, 0), dtype=float)

    if track_index is not None:
        if not (0 <= int(track_index) < len(tracks)):
            raise IndexError(f"track_index {track_index} out of range (0..{len(tracks) - 1})")
        selected_tracks = [tracks[int(track_index)]]
        is_drum = _is_drum_track(selected_tracks[0])
    else:
        if track_mode == "drums":
            selected_tracks = [tr for tr in tracks if _is_drum_track(tr)]
            is_drum = True
        elif track_mode == "pitched":
            selected_tracks = [tr for tr in tracks if not _is_drum_track(tr)]
            is_drum = False
        else:
            selected_tracks = tracks
            is_drum = False

    if track_index is not None and is_drum:
        bMelodicContour = False
        bHarmony = False
        bDissonance = False

    # Build a separate symusic score view for the selected tracks (harmony needs it).
    # We construct a fresh score to avoid mutating the shared `score` object.
    score_sel = symusic.Score(score.ticks_per_quarter)
    score_sel.tempos = score.tempos
    score_sel.time_signatures = score.time_signatures
    for tr in selected_tracks:
        score_sel.tracks.append(tr)

    onsetsAll = {}
    tempoChanges = {0.0: [0.0, 120.0]}

    for t in list(_get_attr(score_sel, "tempos", default=[])):
        tick = int(_get_attr(t, "time", "tick", default=0))
        qpm = float(_get_attr(t, "qpm", "tempo", default=120.0))
        sec = float(tick_to_sec(tick))
        tempoChanges[sec] = [sec, qpm]

    max_end_sec = 0.0
    for tr in selected_tracks:
        for n in list(_get_attr(tr, "notes", default=[])):
            n_start_tick, n_end_tick, pitch, vel = _extract_note_fields(n)
            start_sec = float(tick_to_sec(int(n_start_tick)))
            end_sec = float(tick_to_sec(int(n_end_tick)))
            max_end_sec = max(max_end_sec, end_sec)
            k_sec = int(round(start_sec * _MICRO)) / _MICRO
            note = NoteObj(k_sec, end_sec, int(pitch), int(vel))
            onsetsAll.setdefault(k_sec, []).append(note)

    onsetsAll = dict(sorted(onsetsAll.items()))
    tempoChanges = dict(sorted(tempoChanges.items()))

    totalSamples = int(max_end_sec * sampleRate)
    features = np.zeros((NUM_FEATURES, totalSamples), dtype=float)

    if bOnsetFreq and totalSamples > 0:
        onsetTimes = np.array(list(onsetsAll.keys()), dtype=np.float64)
        if len(onsetTimes) > 1:
            onsetFreq = np.concatenate([[0], 1 / np.diff(onsetTimes)])
        else:
            onsetFreq = np.array([0.0])
        numOnsets = len(onsetTimes)
        sampleIndex = 0
        for j in range(1, numOnsets):
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextOnsetTime = int(onsetTimes[j] * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextOnsetTime:
                features[iOnsetFreq, sampleIndex] = onsetFreq[j - 1]
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
        while sampleIndex <= totalSamples - 1 and numOnsets > 0:
            features[iOnsetFreq, sampleIndex] = onsetFreq[numOnsets - 1]
            sampleIndex += 1

    if bMelodicContour and totalSamples > 0:
        highestPitches = getMelodicLine(onsetsAll)
        sampleIndex = 0
        prevVal = 0
        for key, val in highestPitches.items():
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextNoteTime = int(key * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextNoteTime:
                features[iMelodicContour, sampleIndex] = prevVal
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
            prevVal = val
        while sampleIndex <= totalSamples - 1 and highestPitches:
            features[iMelodicContour, sampleIndex] = prevVal
            sampleIndex += 1
        currIndex = 0
        while currIndex < totalSamples - 1 and features[iMelodicContour, currIndex] == 0:
            currIndex += 1
        if currIndex != totalSamples - 1:
            features[iMelodicContour, :currIndex] = features[iMelodicContour, currIndex]

    if bLoudness and totalSamples > 0:
        loudness = getLoudness(onsetsAll)
        sampleIndex = 0
        prevVal = 0
        for key, val in loudness.items():
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextVal = int(key * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextVal:
                features[iLoudness, sampleIndex] = prevVal
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
            prevVal = val
        while sampleIndex <= totalSamples - 1 and loudness:
            features[iLoudness, sampleIndex] = prevVal
            sampleIndex += 1

    if bTempo and totalSamples > 0:
        sampleIndex = 0
        prevVal = 0
        for _, val in tempoChanges.items():
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextTempoTime = int(val[0] * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextTempoTime:
                features[iTempo, sampleIndex] = prevVal
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
            prevVal = val[1]
        while sampleIndex <= totalSamples - 1 and tempoChanges:
            features[iTempo, sampleIndex] = prevVal
            sampleIndex += 1

    if bHarmony and totalSamples > 0:
        harmonicTension, times = _harmonic_tension_per_track_symusic(
            score_sel, inputFile, window_size=2, end_ratio=1.0, key_changed=False
        )
        numPoints = len(harmonicTension)
        sampleIndex = 0
        for j in range(1, numPoints):
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextVal = int(times[j] * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextVal:
                features[iHarmony, sampleIndex] = harmonicTension[j - 1]
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
        while sampleIndex <= totalSamples - 1 and numPoints > 0:
            features[iHarmony, sampleIndex] = harmonicTension[numPoints - 1]
            sampleIndex += 1

    if bDissonance and totalSamples > 0:
        dissonanceVals = getDissonance(onsetsAll)
        sampleIndex = 0
        prevVal = 0
        for key, val in dissonanceVals.items():
            currSampleTime = int(sampleIndex / sampleRate * 100000)
            nextOnsetTime = int(key * 100000)
            while sampleIndex < totalSamples and currSampleTime < nextOnsetTime:
                features[iDissonance, sampleIndex] = prevVal
                sampleIndex += 1
                currSampleTime = int(sampleIndex / sampleRate * 100000)
            prevVal = val
        while sampleIndex <= totalSamples - 1 and dissonanceVals:
            features[iDissonance, sampleIndex] = prevVal
            sampleIndex += 1

    return features


def compute_track_tension(midi_path: str, track_index: int, is_drum: bool):
    if is_drum:
        weights = FEATURE_WEIGHTS_DRUMS
        bOnsetFreq, bMelodicContour, bLoudness = True, False, True
        bTempo, bHarmony, bDissonance = True, False, False
    else:
        weights = FEATURE_WEIGHTS_PITCHED
        bOnsetFreq, bMelodicContour, bLoudness = True, True, True
        bTempo, bHarmony, bDissonance = True, True, True

    features = extractFeaturesMidi(
        midi_path, SAMPLE_RATE,
        bOnsetFreq, bMelodicContour, bLoudness, bTempo, bHarmony, bDissonance,
        track_index=track_index,
    )

    if features is None or features.shape[1] == 0:
        return np.array([]), np.array([])

    active_flags = [bOnsetFreq, bMelodicContour, bLoudness, bTempo, bHarmony, bDissonance]
    for i, active in enumerate(active_flags):
        if active:
            features[i, :] = normalize(features[i, :])

    featureLen = features.shape[1]
    t_seconds = np.arange(featureLen) / SAMPLE_RATE

    tension = runModel(
        features, [],
        featureList, weights,
        MEMORY_WINDOW_DUR, SAMPLE_RATE,
        ATTENTIONAL_WINDOW_DUR, WINDOW_SHIFT,
        midi_path, MEMORY_WEIGHT, INIT_SLOPE, LAG,
        SLIDER_ONSET,
    )

    return t_seconds, np.asarray(tension, dtype=float)
