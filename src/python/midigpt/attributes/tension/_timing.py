import numpy as np
import symusic


def _ticks_to_seconds(ticks, tempo_ticks, mspq, tpq):
    ticks = np.asarray(ticks, dtype=np.int64)
    tempo_ticks = np.asarray(tempo_ticks, dtype=np.int64)
    mspq = np.asarray(mspq, dtype=np.int64)

    if len(tempo_ticks) == 0 or tempo_ticks[0] != 0:
        tempo_ticks = np.insert(tempo_ticks, 0, 0)
        mspq = np.insert(mspq, 0, 500000)  # 120 bpm default

    cum_sec = np.zeros(len(tempo_ticks), dtype=float)
    for i in range(1, len(tempo_ticks)):
        dticks = tempo_ticks[i] - tempo_ticks[i - 1]
        sec_per_tick = (mspq[i - 1] / 1e6) / tpq
        cum_sec[i] = cum_sec[i - 1] + dticks * sec_per_tick

    seg = np.searchsorted(tempo_ticks, ticks, side="right") - 1
    seg = np.clip(seg, 0, len(tempo_ticks) - 1)
    sec_per_tick = (mspq[seg] / 1e6) / tpq
    return cum_sec[seg] + (ticks - tempo_ticks[seg]) * sec_per_tick


def _load_symusic_maps(midi_path):
    s = symusic.Score(midi_path, ttype="tick")
    if hasattr(s, "sort"):
        s.sort()

    tpq = int(getattr(s, "ticks_per_quarter", getattr(s, "tpq", 480)))

    tempos = s.tempos.numpy()
    tempo_ticks = np.asarray(tempos.get("time", [0]), dtype=np.int64)

    if "mspq" in tempos:
        mspq = np.asarray(tempos["mspq"], dtype=np.int64)
    else:
        qpm = np.asarray(tempos.get("qpm", tempos.get("tempo", [120.0])), dtype=np.float64)
        qpm[qpm <= 0] = 120.0
        mspq = np.asarray(np.round(60_000_000.0 / qpm), dtype=np.int64)

    if len(tempo_ticks) == 0 or tempo_ticks[0] != 0:
        tempo_ticks = np.insert(tempo_ticks, 0, 0)
        mspq = np.insert(mspq, 0, 500000)

    ts = s.time_signatures.numpy()
    if not all(k in ts for k in ("time", "numerator", "denominator")):
        ts_ticks = np.array([0], dtype=np.int64)
        nums = np.array([4], dtype=np.int64)
        dens = np.array([4], dtype=np.int64)
    else:
        ts_ticks = np.asarray(ts["time"], dtype=np.int64)
        nums = np.asarray(ts["numerator"], dtype=np.int64)
        dens = np.asarray(ts["denominator"], dtype=np.int64)
        if len(ts_ticks) == 0 or ts_ticks[0] != 0:
            ts_ticks = np.insert(ts_ticks, 0, 0)
            nums = np.insert(nums, 0, 4)
            dens = np.insert(dens, 0, 4)

    end_tick = int(s.end())
    return tpq, tempo_ticks, mspq, ts_ticks, nums, dens, end_tick


def interval_level_average(tension_arr, sample_times_sec, interval_starts_sec):
    interval_starts_sec = np.asarray(interval_starts_sec, dtype=float)
    if len(interval_starts_sec) < 2:
        return np.array([]), np.array([])

    interval_ids = np.searchsorted(interval_starts_sec[1:], sample_times_sec, side="right")
    n = len(interval_starts_sec) - 1

    vals = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)

    for i, idx in enumerate(interval_ids):
        if 0 <= idx < n:
            vals[idx] += float(tension_arr[i])
            counts[idx] += 1

    counts[counts == 0] = 1
    beat_numbers = np.arange(1, n + 1)
    beat_tension = vals / counts
    return beat_numbers, beat_tension


def get_bar_starts_seconds(midi_path):
    tpq, tempo_ticks, mspq, ts_ticks, nums, dens, end_tick = _load_symusic_maps(midi_path)
    bar_ticks = []
    for i in range(len(ts_ticks)):
        start = int(ts_ticks[i])
        end = int(ts_ticks[i + 1]) if i + 1 < len(ts_ticks) else end_tick
        num = int(nums[i])
        den = int(dens[i])
        ticks_per_bar = round(tpq * (4.0 / den) * num)
        if ticks_per_bar <= 0:
            continue
        t = start
        while t < end:
            bar_ticks.append(t)
            t += ticks_per_bar
    bar_ticks = np.unique(np.asarray(bar_ticks, dtype=np.int64))
    return _ticks_to_seconds(bar_ticks, tempo_ticks, mspq, tpq)
