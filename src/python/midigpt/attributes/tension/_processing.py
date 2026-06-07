import numpy as np

_INTERVAL_DISSONANCE = {
    0: 0.0, 1: 0.85, 2: 0.4, 3: 0.255, 4: 0.225, 5: 0.15,
    6: 0.275, 7: 0.075, 8: 0.275, 9: 0.175, 10: 0.225, 11: 0.4,
}


def normal_round(num, ndigits=0):
    if ndigits == 0:
        return int(num + 0.5)
    digit_value = 10 ** ndigits
    return int(num * digit_value + 0.5) / digit_value


def normalize(arr, filterLen=0):
    out = np.array(arr, dtype=float)
    stdev = np.std(out)
    if stdev != 0:
        return (out - np.mean(out)) / stdev
    return np.zeros(len(arr))


def calculateChordDissonance12tet(notes):
    total = 0.0
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            interval = abs(notes[i] - notes[j]) % 12
            total += _INTERVAL_DISSONANCE[interval]
    return total
