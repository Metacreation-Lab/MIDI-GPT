from midigpt.attributes.tension._processing import calculateChordDissonance12tet

SIXTEENTH = 0.25
EIGHTH = 0.5
QUARTER = 1
HALF = 2
DOTTED_EIGHTH = 0.75


class NoteObj:
    def __init__(self, onset, endTime, pitch, velocity, isRest=False, isTied=False):
        self.onset = float(onset)
        self.endTime = float(endTime)
        self.pitch = int(pitch)
        self.velocity = int(velocity)
        self.durationInSec = self.endTime - self.onset
        self.isRest = isRest
        self.isTied = isTied

    def print(self):
        print(
            "Onset: ", f"{self.onset:.2f}",
            " End: ", f"{self.endTime:.2f}",
            " Pitch: ", self.pitch,
            " Vel: ", self.velocity,
        )


def copy(note):
    return NoteObj(note.onset, note.endTime, note.pitch, note.velocity, note.isRest, note.isTied)


def addOnset(note, newOnset):
    note.onset += newOnset
    note.endTime += newOnset
    note.durationInSec = note.endTime - note.onset


def removeRests(noteList):
    i = len(noteList) - 1
    while i >= 0:
        if noteList[i].isRest:
            noteList.pop(i)
        i -= 1


def mergeTiedNotes(noteList):
    i = len(noteList) - 1
    while i > 0:
        if noteList[i].isTied:
            prevNote = noteList[i - 1]
            if prevNote.pitch == noteList[i].pitch:
                updateEndTime(prevNote, noteList[i].endTime)
                noteList.pop(i)
        i -= 1


def updateEndTime(note, newEndTime):
    note.endTime = float(newEndTime)
    note.durationInSec = note.endTime - note.onset


def newNotebyDur(onset, dur, pitch, velocity, isRest=False, isTied=False):
    return NoteObj(float(onset), float(onset) + float(dur), pitch, velocity, isRest, isTied)


def note_from_symusic(sym_note):
    return NoteObj(sym_note.start, sym_note.end, sym_note.pitch, sym_note.velocity)


def add_note_to_onsets(onsetsAll, onset, end, pitch, velocity, round_decimals=6):
    k = round(float(onset), round_decimals)
    n = NoteObj(k, float(end), int(pitch), int(velocity))
    if k in onsetsAll:
        onsetsAll[k].append(n)
    else:
        onsetsAll[k] = [n]


def getListofPitches(noteList):
    return [n.pitch for n in noteList]


def getHighestPitch(noteList):
    currMax = -1
    for note in noteList:
        if note.pitch > currMax:
            currMax = note.pitch
    return currMax


def getHighestVelocity(noteList):
    currMax = -1
    index = 0
    for i, note in enumerate(noteList):
        if note.velocity > currMax:
            currMax = note.velocity
            index = i
    return currMax, index


def getHighestNote(noteList):
    currMax = -1
    onset = pitch = endTime = velocity = -1
    for note in noteList:
        if note.pitch > currMax:
            onset = note.onset
            pitch = note.pitch
            endTime = note.endTime
            velocity = note.velocity
            currMax = note.pitch
    return NoteObj(onset, endTime, pitch, velocity)


def getMelodicLine(onsetsAll):
    minOnsetDiff = 0.01
    prevPitch = -1
    prevEndTime = -1
    melodicLine = {}
    for key in sorted(onsetsAll.keys()):
        noteList = onsetsAll[key]
        currHighest = getHighestNote(noteList)
        currPitch = currHighest.pitch
        currEndTime = currHighest.endTime
        if (prevEndTime - key < minOnsetDiff) or (prevEndTime > key and currPitch > prevPitch):
            melodicLine[key] = currPitch
            prevPitch = currPitch
            prevEndTime = currEndTime
    return melodicLine


def getLoudness(onsetsAll):
    SCALE_FACTOR = 0.1
    loudness = {}
    for key in sorted(onsetsAll.keys()):
        noteList = onsetsAll[key]
        highestVelocity, index = getHighestVelocity(noteList)
        totalVelocity = 0
        for i, n in enumerate(noteList):
            if i == index:
                totalVelocity += n.velocity
            else:
                totalVelocity += SCALE_FACTOR * n.velocity
        loudness[key] = totalVelocity
    return loudness


def getDissonance(onsetsAll):
    dissonanceVals = {}
    for key in sorted(onsetsAll.keys()):
        chordPitches = getListofPitches(onsetsAll[key])
        dissonanceVals[key] = calculateChordDissonance12tet(chordPitches)
    return dissonanceVals
