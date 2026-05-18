#pragma once

namespace midigpt {

enum class TokenType {
    PieceStart = 0,
    NoteOnset = 1,
    NoteOffset = 2,
    NotePitch = 3,
    NonPitch = 4,
    Velocity = 5,
    TimeDelta = 6,
    TimeAbsolutePos = 7,
    Instrument = 8,
    Bar = 9,
    BarEnd = 10,
    Track = 11,
    TrackEnd = 12,
    DrumTrack = 13,
    FillIn = 14,
    FillInPlaceholder = 15,
    FillInStart = 16,
    FillInEnd = 17,
    Header = 18,
    VelocityLevel = 19,
    Genre = 20,
    NoteDensity = 21,       // TOKEN_DENSITY_LEVEL
    TimeSig = 22,           // TOKEN_TIME_SIGNATURE
    Segment = 23,
    SegmentEnd = 24,
    SegmentFillIn = 25,
    NoteDuration = 26,
    AvPolyphony = 27,
    MinPolyphony = 28,
    MaxPolyphony = 29,
    MinNoteDuration = 30,
    MaxNoteDuration = 31,
    NumBars = 32,
    MinPolyphonyHard = 33,
    MaxPolyphonyHard = 34,
    MinNoteDurationHard = 35,
    MaxNoteDurationHard = 36,
    RestPercentage = 37,
    PitchClass = 38,
    PitchClassCount = 39,
    BarLevelOnsetDensity = 40,
    BarLevelOnsetPolyphonyMin = 41,
    BarLevelOnsetPolyphonyMax = 42,
    TrackLevelOnsetDensity = 43,
    TrackLevelOnsetPolyphonyMin = 44,
    TrackLevelOnsetPolyphonyMax = 45,
    TrackLevelOnsetDensityMin = 46,
    TrackLevelOnsetDensityMax = 47,
    TrackLevelPitchRangeMin = 48,
    TrackLevelPitchRangeMax = 49,
    KeySignature = 50,
    BarLevelPitchClassSet = 51,
    TrackLevelSilenceProportionMin = 52,
    TrackLevelSilenceProportionMax = 53,
    ValenceSpotify = 54,
    EnergySpotify = 55,
    DanceabilitySpotify = 56,
    Danceability = 57,
    Tension = 58,           // TOKEN_BAR_LEVEL_TENSION
    ContainsNoteDurationThirtySecond = 59,
    ContainsNoteDurationSixteenth = 60,
    ContainsNoteDurationEighth = 61,
    ContainsNoteDurationQuarter = 62,
    ContainsNoteDurationHalf = 63,
    ContainsNoteDurationWhole = 64,
    WnbdSyncopation = 65,
    Repetition = 66,
    Delta = 68,
    DeltaDirection = 69,
    None = 70,
    MaskBar = 71,
    TensionDrum = 77,       // TOKEN_BAR_LEVEL_TENSION_DRUM

    // Aliases for refactor names if needed
    OnsetPolyphony = 42,    // Use MaxPolyphony as alias for OnsetPolyphony
    PitchRange = 49,
    NoteDurationDist = 26,
    SilenceProportion = 53,
    PitchClassSet = 51,
    PieceEnd = 70           // Use None for PieceEnd if not present
};

enum class TrackType { Melodic = 0, Drum = 1 };

// Used by attribute constraints: ANY means unconstrained
enum class BooleanEnum { Any = 0, False = 1, True = 2 };

} // namespace midigpt
