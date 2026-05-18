#pragma once

// midi_types.h — replaces protobuf-generated midi.pb.h
// Plain C++ structs + nlohmann/json serializers.
// JSON keys are snake_case throughout (matching what Python sends/expects).

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace midi {

using json = nlohmann::json;

// ── Enums (track_type.proto) ──────────────────────────────────────────────────

enum TRACK_TYPE {
    AUX_DRUM_TRACK    = 8,
    AUX_INST_TRACK    = 9,
    STANDARD_TRACK    = 10,
    STANDARD_DRUM_TRACK = 11,
    STANDARD_BOTH     = 12,
    NUM_TRACK_TYPES   = 16,
};

// ── Enums (enum.proto) ────────────────────────────────────────────────────────

enum TOKEN_TYPE {
    TOKEN_PIECE_START = 0,
    TOKEN_NOTE_ONSET = 1,
    TOKEN_NOTE_OFFSET = 2,
    TOKEN_PITCH = 3,
    TOKEN_NON_PITCH = 4,
    TOKEN_VELOCITY = 5,
    TOKEN_TIME_DELTA = 6,
    TOKEN_TIME_ABSOLUTE_POS = 7,
    TOKEN_INSTRUMENT = 8,
    TOKEN_BAR = 9,
    TOKEN_BAR_END = 10,
    TOKEN_TRACK = 11,
    TOKEN_TRACK_END = 12,
    TOKEN_DRUM_TRACK = 13,
    TOKEN_FILL_IN = 14,
    TOKEN_FILL_IN_PLACEHOLDER = 15,
    TOKEN_FILL_IN_START = 16,
    TOKEN_FILL_IN_END = 17,
    TOKEN_HEADER = 18,
    TOKEN_VELOCITY_LEVEL = 19,
    TOKEN_GENRE = 20,
    TOKEN_DENSITY_LEVEL = 21,
    TOKEN_TIME_SIGNATURE = 22,
    TOKEN_SEGMENT = 23,
    TOKEN_SEGMENT_END = 24,
    TOKEN_SEGMENT_FILL_IN = 25,
    TOKEN_NOTE_DURATION = 26,
    TOKEN_AV_POLYPHONY = 27,
    TOKEN_MIN_POLYPHONY = 28,
    TOKEN_MAX_POLYPHONY = 29,
    TOKEN_MIN_NOTE_DURATION = 30,
    TOKEN_MAX_NOTE_DURATION = 31,
    TOKEN_NUM_BARS = 32,
    TOKEN_MIN_POLYPHONY_HARD = 33,
    TOKEN_MAX_POLYPHONY_HARD = 34,
    TOKEN_MIN_NOTE_DURATION_HARD = 35,
    TOKEN_MAX_NOTE_DURATION_HARD = 36,
    TOKEN_REST_PERCENTAGE = 37,
    TOKEN_PITCH_CLASS = 38,
    TOKEN_PITCH_CLASS_COUNT = 39,
    TOKEN_BAR_LEVEL_ONSET_DENSITY = 40,
    TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MIN = 41,
    TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MAX = 42,
    TOKEN_TRACK_LEVEL_ONSET_DENSITY = 43,
    TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MIN = 44,
    TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MAX = 45,
    TOKEN_TRACK_LEVEL_ONSET_DENSITY_MIN = 46,
    TOKEN_TRACK_LEVEL_ONSET_DENSITY_MAX = 47,
    TOKEN_TRACK_LEVEL_PITCH_RANGE_MIN = 48,
    TOKEN_TRACK_LEVEL_PITCH_RANGE_MAX = 49,
    TOKEN_KEY_SIGNATURE = 50,
    TOKEN_BAR_LEVEL_PITCH_CLASS_SET = 51,
    TOKEN_TRACK_LEVEL_SILENCE_PROPORTION_MIN = 52,
    TOKEN_TRACK_LEVEL_SILENCE_PROPORTION_MAX = 53,
    TOKEN_VALENCE_SPOTIFY = 54,
    TOKEN_ENERGY_SPOTIFY = 55,
    TOKEN_DANCEABILITY_SPOTIFY = 56,
    TOKEN_DANCEABILITY = 57,
    TOKEN_BAR_LEVEL_TENSION = 58,
    TOKEN_CONTAINS_NOTE_DURATION_THIRTY_SECOND = 59,
    TOKEN_CONTAINS_NOTE_DURATION_SIXTEENTH = 60,
    TOKEN_CONTAINS_NOTE_DURATION_EIGHTH = 61,
    TOKEN_CONTAINS_NOTE_DURATION_QUARTER = 62,
    TOKEN_CONTAINS_NOTE_DURATION_HALF = 63,
    TOKEN_CONTAINS_NOTE_DURATION_WHOLE = 64,
    TOKEN_WNBD_SYNCOPATION = 65,
    TOKEN_REPETITION = 66,
    TOKEN_DELTA = 68,
    TOKEN_DELTA_DIRECTION = 69,
    TOKEN_NONE = 70,
    TOKEN_MASK_BAR = 71,
    TOKEN_BAR_LEVEL_TENSION_DRUM = 77,
};

enum ATTRIBUTE_CONTROL_TYPE {
    ATTRIBUTE_CONTROL_NOTE_DENSITY = 0,
    ATTRIBUTE_CONTROL_PITCH_CLASS_COUNT = 1,
    ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_POLYPHONY = 2,
    ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_DENSITY = 3,
    ATTRIBUTE_CONTROL_PITCH_RANGE = 4,
    ATTRIBUTE_CONTROL_KEY_SIGNATURE = 5,
    ATTRIBUTE_CONTROL_BAR_LEVEL_PITCH_CLASS_SET = 6,
    ATTRIBUTE_CONTROL_GENRE = 7,
    ATTRIBUTE_CONTROL_TRACK_LEVEL_SILENCE_PROPORTION = 8,
    ATTRIBUTE_CONTROL_POLYPHONY_QUANTILE = 9,
    ATTRIBUTE_CONTROL_NOTE_DURATION_QUANTILE = 10,
    ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_DENSITY = 11,
    ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_POLYPHONY = 12,
    ATTRIBUTE_CONTROL_VALENCE_SPOTIFY = 13,
    ATTRIBUTE_CONTROL_ENERGY_SPOTIFY = 14,
    ATTRIBUTE_CONTROL_DANCEABILITY_SPOTIFY = 15,
    ATTRIBUTE_CONTROL_DANCEABILITY = 16,
    ATTRIBUTE_CONTROL_TENSION = 17,
    ATTRIBUTE_CONTROL_TRACK_LEVEL_NOTE_DURATION = 18,
    ATTRIBUTE_CONTROL_WNBD_SYNCOPATION = 19,
    ATTRIBUTE_CONTROL_REPETITION = 20,
    ATTRIBUTE_CONTROL_TENSION_DRUM = 21,
    ATTRIBUTE_CONTROL_END = 22,
};

enum GenreMusicmap {
    GENRE_MUSICMAP_ANY = 0,
    GENRE_MUSICMAP_ALTERNATIVE_ROCK = 1,
    GENRE_MUSICMAP_AMBIENT = 2,
    GENRE_MUSICMAP_BLUES = 3,
    GENRE_MUSICMAP_BREAKBEAT = 4,
    GENRE_MUSICMAP_CLASSICAL = 5,
    GENRE_MUSICMAP_CLASSIC_ROCK = 6,
    GENRE_MUSICMAP_CONTEMPORARY_ROCK = 7,
    GENRE_MUSICMAP_COUNTRY = 8,
    GENRE_MUSICMAP_DRUM_N_BASS = 9,
    GENRE_MUSICMAP_FOLK = 10,
    GENRE_MUSICMAP_GOSPEL = 11,
    GENRE_MUSICMAP_HARDCORE_PUNK = 12,
    GENRE_MUSICMAP_HARDCORE_TECHNO = 13,
    GENRE_MUSICMAP_HEAVY_METAL = 14,
    GENRE_MUSICMAP_HIP_HOP = 15,
    GENRE_MUSICMAP_HOUSE = 16,
    GENRE_MUSICMAP_INDUSTRIAL = 17,
    GENRE_MUSICMAP_JAZZ = 18,
    GENRE_MUSICMAP_LATIN = 19,
    GENRE_MUSICMAP_POP = 20,
    GENRE_MUSICMAP_PUNK = 21,
    GENRE_MUSICMAP_PUNK_ROCK = 22,
    GENRE_MUSICMAP_RANDB = 23,
    GENRE_MUSICMAP_REGGAE = 24,
    GENRE_MUSICMAP_ROCK_N_ROLL = 25,
    GENRE_MUSICMAP_TECHNO = 26,
    GENRE_MUSICMAP_TRANCE = 27,
    GENRE_MUSICMAP_UTILITY = 28,
    GENRE_MUSICMAP_WORLD = 29,
    GENRE_MUSICMAP_NONE = 30,
};

enum GM_CATEGORY {
    GM_CATEGORY_MONO = 0,
    GM_CATEGORY_POLY = 1,
    GM_CATEGORY_SOUND_FX = 2,
    GM_CATEGORY_PERC = 3,
};

enum GM_TYPE {
    any = 0, piano = 1, chromatic_perc = 2, organ = 3, guitar = 4,
    bass = 5, strings = 6, ensemble = 7, brass = 8, reed = 9,
    pipe = 10, synth_lead = 11, synth_pad = 12, synth_effects = 13,
    ethnic = 14, percussive = 15, sound_fx = 16, no_drums = 17, drums = 18,
    acoustic_grand_piano = 19, bright_acoustic_piano = 20, electric_grand_piano = 21,
    honky_tonk_piano = 22, electric_piano_1 = 23, electric_piano_2 = 24,
    harpsichord = 25, clavi = 26, celesta = 27, glockenspiel = 28,
    music_box = 29, vibraphone = 30, marimba = 31, xylophone = 32,
    tubular_bells = 33, dulcimer = 34, drawbar_organ = 35, percussive_organ = 36,
    rock_organ = 37, church_organ = 38, reed_organ = 39, accordion = 40,
    harmonica = 41, tango_accordion = 42, acoustic_guitar_nylon = 43,
    acoustic_guitar_steel = 44, electric_guitar_jazz = 45,
    electric_guitar_clean = 46, electric_guitar_muted = 47,
    overdriven_guitar = 48, distortion_guitar = 49, guitar_harmonics = 50,
    acoustic_bass = 51, electric_bass_finger = 52, electric_bass_pick = 53,
    fretless_bass = 54, slap_bass_1 = 55, slap_bass_2 = 56,
    synth_bass_1 = 57, synth_bass_2 = 58, violin = 59, viola = 60,
    cello = 61, contrabass = 62, tremolo_strings = 63, pizzicato_strings = 64,
    orchestral_harp = 65, timpani = 66, string_ensemble_1 = 67,
    string_ensemble_2 = 68, synth_strings_1 = 69, synth_strings_2 = 70,
    choir_aahs = 71, voice_oohs = 72, synth_voice = 73, orchestra_hit = 74,
    trumpet = 75, trombone = 76, tuba = 77, muted_trumpet = 78,
    french_horn = 79, brass_section = 80, synth_brass_1 = 81, synth_brass_2 = 82,
    soprano_sax = 83, alto_sax = 84, tenor_sax = 85, baritone_sax = 86,
    oboe = 87, english_horn = 88, bassoon = 89, clarinet = 90,
    piccolo = 91, flute = 92, recorder = 93, pan_flute = 94,
    blown_bottle = 95, shakuhachi = 96, whistle = 97, ocarina = 98,
    lead_1_square = 99, lead_2_sawtooth = 100, lead_3_calliope = 101,
    lead_4_chiff = 102, lead_5_charang = 103, lead_6_voice = 104,
    lead_7_fifths = 105, lead_8_bass__lead = 106,
    pad_1_new_age = 107, pad_2_warm = 108, pad_3_polysynth = 109,
    pad_4_choir = 110, pad_5_bowed = 111, pad_6_metallic = 112,
    pad_7_halo = 113, pad_8_sweep = 114,
    fx_1_rain = 115, fx_2_soundtrack = 116, fx_3_crystal = 117,
    fx_4_atmosphere = 118, fx_5_brightness = 119, fx_6_goblins = 120,
    fx_7_echoes = 121, fx_8_sci_fi = 122,
    sitar = 123, banjo = 124, shamisen = 125, koto = 126, kalimba = 127,
    bag_pipe = 128, fiddle = 129, shanai = 130,
    tinkle_bell = 131, agogo = 132, steel_drums = 133, woodblock = 134,
    taiko_drum = 135, melodic_tom = 136, synth_drum = 137,
    reverse_cymbal = 138, guitar_fret_noise = 139, breath_noise = 140,
    seashore = 141, bird_tweet = 142, telephone_ring = 143,
    helicopter = 144, applause = 145, gunshot = 146,
    drum_0 = 147, drum_1 = 148, drum_2 = 149, drum_3 = 150, drum_4 = 151,
    drum_5 = 152, drum_6 = 153, drum_7 = 154, drum_8 = 155, drum_9 = 156,
    drum_10 = 157, drum_11 = 158, drum_12 = 159, drum_13 = 160, drum_14 = 161,
    drum_15 = 162, drum_16 = 163, drum_17 = 164, drum_18 = 165, drum_19 = 166,
    drum_20 = 167, drum_21 = 168, drum_22 = 169, drum_23 = 170, drum_24 = 171,
    drum_25 = 172, drum_26 = 173, drum_27 = 174, drum_28 = 175, drum_29 = 176,
    drum_30 = 177, drum_31 = 178, drum_32 = 179, drum_33 = 180, drum_34 = 181,
    drum_35 = 182, drum_36 = 183, drum_37 = 184, drum_38 = 185, drum_39 = 186,
    drum_40 = 187, drum_41 = 188, drum_42 = 189, drum_43 = 190, drum_44 = 191,
    drum_45 = 192, drum_46 = 193, drum_47 = 194, drum_48 = 195, drum_49 = 196,
    drum_50 = 197, drum_51 = 198, drum_52 = 199, drum_53 = 200, drum_54 = 201,
    drum_55 = 202, drum_56 = 203, drum_57 = 204, drum_58 = 205, drum_59 = 206,
    drum_60 = 207, drum_61 = 208, drum_62 = 209, drum_63 = 210, drum_64 = 211,
    drum_65 = 212, drum_66 = 213, drum_67 = 214, drum_68 = 215, drum_69 = 216,
    drum_70 = 217, drum_71 = 218, drum_72 = 219, drum_73 = 220, drum_74 = 221,
    drum_75 = 222, drum_76 = 223, drum_77 = 224, drum_78 = 225, drum_79 = 226,
    drum_80 = 227, drum_81 = 228, drum_82 = 229, drum_83 = 230, drum_84 = 231,
    drum_85 = 232, drum_86 = 233, drum_87 = 234, drum_88 = 235, drum_89 = 236,
    drum_90 = 237, drum_91 = 238, drum_92 = 239, drum_93 = 240, drum_94 = 241,
    drum_95 = 242, drum_96 = 243, drum_97 = 244, drum_98 = 245, drum_99 = 246,
    drum_100 = 247, drum_101 = 248, drum_102 = 249, drum_103 = 250, drum_104 = 251,
    drum_105 = 252, drum_106 = 253, drum_107 = 254, drum_108 = 255, drum_109 = 256,
    drum_110 = 257, drum_111 = 258, drum_112 = 259, drum_113 = 260, drum_114 = 261,
    drum_115 = 262, drum_116 = 263, drum_117 = 264, drum_118 = 265, drum_119 = 266,
    drum_120 = 267, drum_121 = 268, drum_122 = 269, drum_123 = 270, drum_124 = 271,
    drum_125 = 272, drum_126 = 273, drum_127 = 274,
};

// Level enums (from midi.proto)
enum PolyphonyLevel {
    POLYPHONY_ANY=0, POLYPHONY_ONE=1, POLYPHONY_TWO=2, POLYPHONY_THREE=3,
    POLYPHONY_FOUR=4, POLYPHONY_FIVE=5, POLYPHONY_SIX=6,
};
enum NoteDurationLevel {
    DURATION_ANY=0, DURATION_THIRTY_SECOND=1, DURATION_SIXTEENTH=2,
    DURATION_EIGHTH=3, DURATION_QUARTER=4, DURATION_HALF=5, DURATION_WHOLE=6,
};
enum DensityLevel {
    DENSITY_ANY=0, DENSITY_ONE=1, DENSITY_TWO=2, DENSITY_THREE=3,
    DENSITY_FOUR=4, DENSITY_FIVE=5, DENSITY_SIX=6, DENSITY_SEVEN=7,
    DENSITY_EIGHT=8, DENSITY_NINE=9, DENSITY_TEN=10,
};
enum BarLevelOnsetDensityLevel {
    BAR_LEVEL_ONSET_DENSITY_ANY=0,  BAR_LEVEL_ONSET_DENSITY_ZERO=1,
    BAR_LEVEL_ONSET_DENSITY_ONE=2,  BAR_LEVEL_ONSET_DENSITY_TWO=3,
    BAR_LEVEL_ONSET_DENSITY_THREE=4, BAR_LEVEL_ONSET_DENSITY_FOUR=5,
    BAR_LEVEL_ONSET_DENSITY_FIVE=6, BAR_LEVEL_ONSET_DENSITY_SIX=7,
    BAR_LEVEL_ONSET_DENSITY_SEVEN=8, BAR_LEVEL_ONSET_DENSITY_EIGHT=9,
    BAR_LEVEL_ONSET_DENSITY_NINE=10, BAR_LEVEL_ONSET_DENSITY_TEN=11,
    BAR_LEVEL_ONSET_DENSITY_ELEVEN=12, BAR_LEVEL_ONSET_DENSITY_TWELVE=13,
    BAR_LEVEL_ONSET_DENSITY_THIRTEEN=14, BAR_LEVEL_ONSET_DENSITY_FOURTEEN=15,
    BAR_LEVEL_ONSET_DENSITY_FIFTEEN=16, BAR_LEVEL_ONSET_DENSITY_SIXTEEN=17,
};
enum BarLevelOnsetPolyphonyLevel {
    BAR_LEVEL_ONSET_POLYPHONY_ANY=0, BAR_LEVEL_ONSET_POLYPHONY_ONE=1,
    BAR_LEVEL_ONSET_POLYPHONY_TWO=2, BAR_LEVEL_ONSET_POLYPHONY_THREE=3,
    BAR_LEVEL_ONSET_POLYPHONY_FOUR=4, BAR_LEVEL_ONSET_POLYPHONY_FIVE=5,
    BAR_LEVEL_ONSET_POLYPHONY_SIX=6,
};
enum SilenceProportionLevel {
    SILENCE_PROPORTION_LEVEL_ANY=0, SILENCE_PROPORTION_LEVEL_ONE=1,
    SILENCE_PROPORTION_LEVEL_TWO=2, SILENCE_PROPORTION_LEVEL_THREE=3,
    SILENCE_PROPORTION_LEVEL_FOUR=4, SILENCE_PROPORTION_LEVEL_FIVE=5,
    SILENCE_PROPORTION_LEVEL_SIX=6, SILENCE_PROPORTION_LEVEL_SEVEN=7,
    SILENCE_PROPORTION_LEVEL_EIGHT=8, SILENCE_PROPORTION_LEVEL_NINE=9,
    SILENCE_PROPORTION_LEVEL_TEN=10,
};
enum BooleanLevel { BOOLEAN_ANY=0, BOOLEAN_FALSE=1, BOOLEAN_TRUE=2 };
enum PitchClassCountLevel {
    PITCH_CLASS_COUNT_ANY=0, PITCH_CLASS_COUNT_ONE=1, PITCH_CLASS_COUNT_TWO=2,
    PITCH_CLASS_COUNT_THREE=3, PITCH_CLASS_COUNT_FOUR=4, PITCH_CLASS_COUNT_FIVE=5,
    PITCH_CLASS_COUNT_SIX=6, PITCH_CLASS_COUNT_SEVEN=7, PITCH_CLASS_COUNT_EIGHT=8,
    PITCH_CLASS_COUNT_NINE=9, PITCH_CLASS_COUNT_TEN=10, PITCH_CLASS_COUNT_ELEVEN=11,
    PITCH_CLASS_COUNT_TWELVE=12,
};
enum KeySignatureLevel {
    KEY_SIGNATURE_ANY=0,
    KEY_SIGNATURE_C_MAJOR=1,  KEY_SIGNATURE_Db_MAJOR=2, KEY_SIGNATURE_D_MAJOR=3,
    KEY_SIGNATURE_Eb_MAJOR=4, KEY_SIGNATURE_E_MAJOR=5,  KEY_SIGNATURE_F_MAJOR=6,
    KEY_SIGNATURE_Gb_MAJOR=7, KEY_SIGNATURE_G_MAJOR=8,  KEY_SIGNATURE_Ab_MAJOR=9,
    KEY_SIGNATURE_A_MAJOR=10, KEY_SIGNATURE_Bb_MAJOR=11, KEY_SIGNATURE_B_MAJOR=12,
    KEY_SIGNATURE_C_MINOR=13, KEY_SIGNATURE_Db_MINOR=14, KEY_SIGNATURE_D_MINOR=15,
    KEY_SIGNATURE_Eb_MINOR=16, KEY_SIGNATURE_E_MINOR=17, KEY_SIGNATURE_F_MINOR=18,
    KEY_SIGNATURE_Gb_MINOR=19, KEY_SIGNATURE_G_MINOR=20, KEY_SIGNATURE_Ab_MINOR=21,
    KEY_SIGNATURE_A_MINOR=22, KEY_SIGNATURE_Bb_MINOR=23, KEY_SIGNATURE_B_MINOR=24,
};
enum DecileLevel {
    DECILE_LEVEL_ANY=0, DECILE_LEVEL_ONE=1, DECILE_LEVEL_TWO=2,
    DECILE_LEVEL_THREE=3, DECILE_LEVEL_FOUR=4, DECILE_LEVEL_FIVE=5,
    DECILE_LEVEL_SIX=6, DECILE_LEVEL_SEVEN=7, DECILE_LEVEL_EIGHT=8,
    DECILE_LEVEL_NINE=9, DECILE_LEVEL_TEN=10, DECILE_LEVEL_NONE=11,
};

// ── GM_TYPE string ↔ int lookup ───────────────────────────────────────────────

inline int gm_type_from_string(const std::string &s) {
    static const std::map<std::string, int> tbl = {
        {"any",0},{"piano",1},{"chromatic_perc",2},{"organ",3},{"guitar",4},
        {"bass",5},{"strings",6},{"ensemble",7},{"brass",8},{"reed",9},
        {"pipe",10},{"synth_lead",11},{"synth_pad",12},{"synth_effects",13},
        {"ethnic",14},{"percussive",15},{"sound_fx",16},{"no_drums",17},{"drums",18},
        {"acoustic_grand_piano",19},{"bright_acoustic_piano",20},
        {"electric_grand_piano",21},{"honky_tonk_piano",22},
        {"electric_piano_1",23},{"electric_piano_2",24},
        {"harpsichord",25},{"clavi",26},{"celesta",27},{"glockenspiel",28},
        {"music_box",29},{"vibraphone",30},{"marimba",31},{"xylophone",32},
        {"tubular_bells",33},{"dulcimer",34},{"drawbar_organ",35},
        {"percussive_organ",36},{"rock_organ",37},{"church_organ",38},
        {"reed_organ",39},{"accordion",40},{"harmonica",41},{"tango_accordion",42},
        {"acoustic_guitar_nylon",43},{"acoustic_guitar_steel",44},
        {"electric_guitar_jazz",45},{"electric_guitar_clean",46},
        {"electric_guitar_muted",47},{"overdriven_guitar",48},
        {"distortion_guitar",49},{"guitar_harmonics",50},{"acoustic_bass",51},
        {"electric_bass_finger",52},{"electric_bass_pick",53},{"fretless_bass",54},
        {"slap_bass_1",55},{"slap_bass_2",56},{"synth_bass_1",57},{"synth_bass_2",58},
        {"violin",59},{"viola",60},{"cello",61},{"contrabass",62},
        {"tremolo_strings",63},{"pizzicato_strings",64},{"orchestral_harp",65},
        {"timpani",66},{"string_ensemble_1",67},{"string_ensemble_2",68},
        {"synth_strings_1",69},{"synth_strings_2",70},{"choir_aahs",71},
        {"voice_oohs",72},{"synth_voice",73},{"orchestra_hit",74},
        {"trumpet",75},{"trombone",76},{"tuba",77},{"muted_trumpet",78},
        {"french_horn",79},{"brass_section",80},{"synth_brass_1",81},{"synth_brass_2",82},
        {"soprano_sax",83},{"alto_sax",84},{"tenor_sax",85},{"baritone_sax",86},
        {"oboe",87},{"english_horn",88},{"bassoon",89},{"clarinet",90},
        {"piccolo",91},{"flute",92},{"recorder",93},{"pan_flute",94},
        {"blown_bottle",95},{"shakuhachi",96},{"whistle",97},{"ocarina",98},
        {"lead_1_square",99},{"lead_2_sawtooth",100},{"lead_3_calliope",101},
        {"lead_4_chiff",102},{"lead_5_charang",103},{"lead_6_voice",104},
        {"lead_7_fifths",105},{"lead_8_bass__lead",106},
        {"pad_1_new_age",107},{"pad_2_warm",108},{"pad_3_polysynth",109},
        {"pad_4_choir",110},{"pad_5_bowed",111},{"pad_6_metallic",112},
        {"pad_7_halo",113},{"pad_8_sweep",114},
        {"fx_1_rain",115},{"fx_2_soundtrack",116},{"fx_3_crystal",117},
        {"fx_4_atmosphere",118},{"fx_5_brightness",119},{"fx_6_goblins",120},
        {"fx_7_echoes",121},{"fx_8_sci_fi",122},
        {"sitar",123},{"banjo",124},{"shamisen",125},{"koto",126},{"kalimba",127},
        {"bag_pipe",128},{"fiddle",129},{"shanai",130},
        {"tinkle_bell",131},{"agogo",132},{"steel_drums",133},{"woodblock",134},
        {"taiko_drum",135},{"melodic_tom",136},{"synth_drum",137},
        {"reverse_cymbal",138},{"guitar_fret_noise",139},{"breath_noise",140},
        {"seashore",141},{"bird_tweet",142},{"telephone_ring",143},
        {"helicopter",144},{"applause",145},{"gunshot",146},
    };
    auto it = tbl.find(s);
    return it != tbl.end() ? it->second : 0;
}

// ── Feature structs (midi_internal.proto) ─────────────────────────────────────

struct ContinuousFeature {
    float av_polyphony = 0;
    float av_silence = 0;
    float note_duration = 0;
    float note_duration_norm = 0;
};

struct BarFeatures {
    int onset_density = 0;
    int onset_polyphony_min = 0;
    int onset_polyphony_max = 0;
    std::vector<int> pitch_class_set;
    int tension = 0;
    int tension_drum = 0;
    float tension_raw = 0;
    float tension_drum_raw = 0;
};

struct TrackLevelAttributeControlDistributions {
    std::vector<int> polyphony_quantile;
    std::vector<int> note_duration_quantile;
    std::vector<int> note_density;
    std::vector<int> onset_polyphony;
    std::vector<int> onset_density;
    std::vector<int> note_duration;
    std::vector<float> key_signature_distribution;
    std::vector<float> silence_proportion;
};

struct TrackFeatures {
    int min_pitch = 0;
    int max_pitch = 0;
    float av_polyphony = 0;
    int note_density = 0;
    int note_density_v2 = 0;
    int max_polyphony = 0;
    bool should_prune = false;
    int order = 0;
    float note_duration = 0;
    std::string genre_str;
    int min_polyphony_q = 0;
    int max_polyphony_q = 0;
    int min_note_duration_q = 0;
    int max_note_duration_q = 0;
    std::vector<int> polyphony_distribution;
    float note_density_value = 0;
    int min_polyphony_hard = 0;
    int max_polyphony_hard = 0;
    int min_note_duration_hard = 0;
    int max_note_duration_hard = 0;
    int onset_polyphony_min = 0;
    int onset_polyphony_max = 0;
    int onset_density = 0;
    int onset_density_min = 0;
    int onset_density_max = 0;
    std::vector<int> duration_distribution;
    int genre = 0;
    int note_density_level = 0;
    int contains_note_duration_thirty_second = 0;
    int contains_note_duration_sixteenth = 0;
    int contains_note_duration_eighth = 0;
    int contains_note_duration_quarter = 0;
    int contains_note_duration_half = 0;
    int contains_note_duration_whole = 0;
    std::vector<bool> pitch_classes;
    float rest_percentage = 0;
    int pitch_class_count = 0;
    int key_signature = 0;
    int silence_proportion_min = 0;
    int silence_proportion_max = 0;
    int valence_spotify = 0;
    int energy_spotify = 0;
    int danceability_spotify = 0;
    int danceability = 0;
    int wnbd_syncopation = 0;
    int repetition = 0;
    TrackLevelAttributeControlDistributions attribute_control_distributions;
};

struct PieceFeatures {
    std::string genre;
    int danceability = 0;
};

struct GenreData {
    std::string discogs;
    std::string lastfm;
    std::string tagtraum;
};

struct MetadataLabels {
    int genre = 0;
    float valence_spotify = 0;
    float energy_spotify = 0;
    float danceability_spotify = 0;
    std::vector<float> tension;
    int nomml = 0;
};

struct ValidTrack {
    std::vector<int> tracks;
};

struct Note {
    int start = 0;
    int end = 0;
    int pitch = 0;
    int tick_delta = 0;
};

// ── Core MIDI structs (midi.proto) ────────────────────────────────────────────

struct TempoChange {
    int tick = 0;
    float qpm = 0;
};

struct Event {
    int time = 0;
    int velocity = 0;
    int pitch = 0;
    int internal_instrument = 0;
    int internal_track_type = 0;
    int internal_duration = 0;
    int delta = 0;
};

struct Bar {
    std::vector<int> events;
    int ts_numerator = 4;
    int ts_denominator = 4;
    float internal_beat_length = 0;
    bool internal_has_notes = false;
    std::vector<ContinuousFeature> internal_feature;
    std::vector<BarFeatures> internal_features;
    bool future = false;
};

struct Track {
    std::vector<Bar> bars;
    int instrument = 0;
    int track_type = STANDARD_TRACK;
    std::vector<int> internal_train_types;
    std::vector<TrackFeatures> internal_features;
};

struct Piece {
    std::vector<Track> tracks;
    std::vector<Event> events;
    int resolution = 12;
    int tempo = 0;
    std::vector<int> internal_valid_segments;
    std::vector<uint32_t> internal_valid_tracks;
    int internal_segment_length = 0;
    std::vector<ValidTrack> internal_valid_tracks_v2;
    std::vector<GenreData> internal_genre_data;
    MetadataLabels internal_metadata_labels;
    std::vector<PieceFeatures> internal_features;
    int internal_ticks_per_quarter = 0;
    bool internal_has_time_signatures = false;
    std::vector<TempoChange> tempo_changes;
};

// ── API structs ───────────────────────────────────────────────────────────────

struct StatusBar {
    int ts_numerator = 0;
    int ts_denominator = 0;
    int onset_density = 0;
    int onset_polyphony_min = 0;
    int onset_polyphony_max = 0;
    std::vector<bool> pitch_class_set;
    int tension = 0;
    bool future = false;
    int tension_drum = 0;
};

struct StatusTrack {
    int track_id = 0;
    int track_type = STANDARD_TRACK;
    int instrument = 0;  // GM_TYPE int; parsed from int or string in JSON
    std::vector<bool> selected_bars;
    bool autoregressive = false;
    bool ignore = false;
    int density = 0;
    int min_polyphony_q = 0;
    int max_polyphony_q = 0;
    int min_note_duration_q = 0;
    int max_note_duration_q = 0;
    int onset_polyphony_min = 0;
    int onset_polyphony_max = 0;
    int onset_density = 0;
    int onset_density_min = 0;
    int onset_density_max = 0;
    int min_pitch = 0;
    int max_pitch = 0;
    int genre = 0;
    int silence_proportion_min = 0;
    int silence_proportion_max = 0;
    int note_density_level = 0;
    int contains_note_duration_thirty_second = 0;
    int contains_note_duration_sixteenth = 0;
    int contains_note_duration_eighth = 0;
    int contains_note_duration_quarter = 0;
    int contains_note_duration_half = 0;
    int contains_note_duration_whole = 0;
    int pitch_class_count = 0;
    int key_signature = 0;
    int valence_spotify = 0;
    int energy_spotify = 0;
    int danceability_spotify = 0;
    int danceability = 0;
    int wnbd_syncopation = 0;
    int repetition = 0;
    bool suffix_autoregressive = false;
    int polyphony_hard_limit = 0;
    float temperature = 0;  // 0 = not set (use global)
    std::vector<int> internal_ts_numerators;
    std::vector<int> internal_ts_denominators;
    std::string internal_genre;
    std::vector<StatusBar> bars;
};

struct Status {
    std::vector<StatusTrack> tracks;
    bool decode_final = false;
    bool full_resolution = false;
};

struct HyperParam {
    int tracks_per_step = 1;
    int bars_per_step = 1;
    int model_dim = 4;
    int percentage = 100;
    int batch_size = 1;
    float temperature = 1.0f;
    bool use_per_track_temperature = false;
    int max_steps = 0;
    int polyphony_hard_limit = 0;
    bool shuffle = false;
    bool verbose = false;
    std::string ckpt;
    float mask_top_k = 0;
    int sampling_seed = -1;
    bool internal_skip_preprocess = false;
    bool internal_disable_masking = false;
};

// ── nlohmann/json serializers ─────────────────────────────────────────────────
// All keys are snake_case, matching what Python sends and expects.

// ─ ContinuousFeature ─

inline void from_json(const json &j, ContinuousFeature &x) {
    x.av_polyphony     = j.value("av_polyphony", 0.0f);
    x.av_silence       = j.value("av_silence", 0.0f);
    x.note_duration    = j.value("note_duration", 0.0f);
    x.note_duration_norm = j.value("note_duration_norm", 0.0f);
}
inline void to_json(json &j, const ContinuousFeature &x) {
    j = json{{"av_polyphony",x.av_polyphony},{"av_silence",x.av_silence},
             {"note_duration",x.note_duration},{"note_duration_norm",x.note_duration_norm}};
}

// ─ BarFeatures ─

inline void from_json(const json &j, BarFeatures &x) {
    x.onset_density       = j.value("onset_density", 0);
    x.onset_polyphony_min = j.value("onset_polyphony_min", 0);
    x.onset_polyphony_max = j.value("onset_polyphony_max", 0);
    if (j.contains("pitch_class_set")) x.pitch_class_set = j["pitch_class_set"].get<std::vector<int>>();
    x.tension      = j.value("tension", 0);
    x.tension_drum = j.value("tension_drum", 0);
    x.tension_raw      = j.value("tension_raw", 0.0f);
    x.tension_drum_raw = j.value("tension_drum_raw", 0.0f);
}
inline void to_json(json &j, const BarFeatures &x) {
    j = json{};
    if (x.onset_density)       j["onset_density"] = x.onset_density;
    if (x.onset_polyphony_min) j["onset_polyphony_min"] = x.onset_polyphony_min;
    if (x.onset_polyphony_max) j["onset_polyphony_max"] = x.onset_polyphony_max;
    if (!x.pitch_class_set.empty()) j["pitch_class_set"] = x.pitch_class_set;
    if (x.tension)             j["tension"] = x.tension;
    if (x.tension_drum)        j["tension_drum"] = x.tension_drum;
    if (x.tension_raw != 0)    j["tension_raw"] = x.tension_raw;
    if (x.tension_drum_raw != 0) j["tension_drum_raw"] = x.tension_drum_raw;
}

// ─ TrackLevelAttributeControlDistributions ─

inline void from_json(const json &j, TrackLevelAttributeControlDistributions &x) {
    if (j.contains("polyphony_quantile"))       x.polyphony_quantile       = j["polyphony_quantile"].get<std::vector<int>>();
    if (j.contains("note_duration_quantile"))   x.note_duration_quantile   = j["note_duration_quantile"].get<std::vector<int>>();
    if (j.contains("note_density"))             x.note_density             = j["note_density"].get<std::vector<int>>();
    if (j.contains("onset_polyphony"))          x.onset_polyphony          = j["onset_polyphony"].get<std::vector<int>>();
    if (j.contains("onset_density"))            x.onset_density            = j["onset_density"].get<std::vector<int>>();
    if (j.contains("note_duration"))            x.note_duration            = j["note_duration"].get<std::vector<int>>();
    if (j.contains("key_signature_distribution")) x.key_signature_distribution = j["key_signature_distribution"].get<std::vector<float>>();
    if (j.contains("silence_proportion"))       x.silence_proportion       = j["silence_proportion"].get<std::vector<float>>();
}
inline void to_json(json &j, const TrackLevelAttributeControlDistributions &x) {
    j = json{};
    if (!x.polyphony_quantile.empty())         j["polyphony_quantile"] = x.polyphony_quantile;
    if (!x.note_duration_quantile.empty())     j["note_duration_quantile"] = x.note_duration_quantile;
    if (!x.note_density.empty())               j["note_density"] = x.note_density;
    if (!x.onset_polyphony.empty())            j["onset_polyphony"] = x.onset_polyphony;
    if (!x.onset_density.empty())              j["onset_density"] = x.onset_density;
    if (!x.note_duration.empty())              j["note_duration"] = x.note_duration;
    if (!x.key_signature_distribution.empty()) j["key_signature_distribution"] = x.key_signature_distribution;
    if (!x.silence_proportion.empty())         j["silence_proportion"] = x.silence_proportion;
}

// ─ TrackFeatures ─

inline void from_json(const json &j, TrackFeatures &x) {
    x.min_pitch = j.value("min_pitch", 0);
    x.max_pitch = j.value("max_pitch", 0);
    x.av_polyphony = j.value("av_polyphony", 0.0f);
    x.note_density = j.value("note_density", 0);
    x.note_density_v2 = j.value("note_density_v2", 0);
    x.max_polyphony = j.value("max_polyphony", 0);
    x.should_prune = j.value("should_prune", false);
    x.order = j.value("order", 0);
    x.note_duration = j.value("note_duration", 0.0f);
    x.genre_str = j.value("genre_str", std::string{});
    x.min_polyphony_q = j.value("min_polyphony_q", 0);
    x.max_polyphony_q = j.value("max_polyphony_q", 0);
    x.min_note_duration_q = j.value("min_note_duration_q", 0);
    x.max_note_duration_q = j.value("max_note_duration_q", 0);
    if (j.contains("polyphony_distribution")) x.polyphony_distribution = j["polyphony_distribution"].get<std::vector<int>>();
    x.note_density_value = j.value("note_density_value", 0.0f);
    x.min_polyphony_hard = j.value("min_polyphony_hard", 0);
    x.max_polyphony_hard = j.value("max_polyphony_hard", 0);
    x.min_note_duration_hard = j.value("min_note_duration_hard", 0);
    x.max_note_duration_hard = j.value("max_note_duration_hard", 0);
    x.onset_polyphony_min = j.value("onset_polyphony_min", 0);
    x.onset_polyphony_max = j.value("onset_polyphony_max", 0);
    x.onset_density = j.value("onset_density", 0);
    x.onset_density_min = j.value("onset_density_min", 0);
    x.onset_density_max = j.value("onset_density_max", 0);
    if (j.contains("duration_distribution")) x.duration_distribution = j["duration_distribution"].get<std::vector<int>>();
    x.genre = j.value("genre", 0);
    x.note_density_level = j.value("note_density_level", 0);
    x.contains_note_duration_thirty_second = j.value("contains_note_duration_thirty_second", 0);
    x.contains_note_duration_sixteenth = j.value("contains_note_duration_sixteenth", 0);
    x.contains_note_duration_eighth = j.value("contains_note_duration_eighth", 0);
    x.contains_note_duration_quarter = j.value("contains_note_duration_quarter", 0);
    x.contains_note_duration_half = j.value("contains_note_duration_half", 0);
    x.contains_note_duration_whole = j.value("contains_note_duration_whole", 0);
    if (j.contains("pitch_classes")) x.pitch_classes = j["pitch_classes"].get<std::vector<bool>>();
    x.rest_percentage = j.value("rest_percentage", 0.0f);
    x.pitch_class_count = j.value("pitch_class_count", 0);
    x.key_signature = j.value("key_signature", 0);
    x.silence_proportion_min = j.value("silence_proportion_min", 0);
    x.silence_proportion_max = j.value("silence_proportion_max", 0);
    x.valence_spotify = j.value("valence_spotify", 0);
    x.energy_spotify = j.value("energy_spotify", 0);
    x.danceability_spotify = j.value("danceability_spotify", 0);
    x.danceability = j.value("danceability", 0);
    x.wnbd_syncopation = j.value("wnbd_syncopation", 0);
    x.repetition = j.value("repetition", 0);
    if (j.contains("attribute_control_distributions"))
        x.attribute_control_distributions = j["attribute_control_distributions"].get<TrackLevelAttributeControlDistributions>();
}
inline void to_json(json &j, const TrackFeatures &x) {
    j = json{};
    if (x.min_pitch)   j["min_pitch"] = x.min_pitch;
    if (x.max_pitch)   j["max_pitch"] = x.max_pitch;
    if (x.av_polyphony != 0) j["av_polyphony"] = x.av_polyphony;
    if (x.note_density) j["note_density"] = x.note_density;
    if (x.note_density_v2) j["note_density_v2"] = x.note_density_v2;
    if (x.max_polyphony) j["max_polyphony"] = x.max_polyphony;
    if (x.should_prune) j["should_prune"] = x.should_prune;
    if (x.order)        j["order"] = x.order;
    if (x.note_duration != 0) j["note_duration"] = x.note_duration;
    if (!x.genre_str.empty()) j["genre_str"] = x.genre_str;
    if (x.min_polyphony_q) j["min_polyphony_q"] = x.min_polyphony_q;
    if (x.max_polyphony_q) j["max_polyphony_q"] = x.max_polyphony_q;
    if (x.min_note_duration_q) j["min_note_duration_q"] = x.min_note_duration_q;
    if (x.max_note_duration_q) j["max_note_duration_q"] = x.max_note_duration_q;
    if (!x.polyphony_distribution.empty()) j["polyphony_distribution"] = x.polyphony_distribution;
    if (x.note_density_value != 0) j["note_density_value"] = x.note_density_value;
    if (x.onset_polyphony_min) j["onset_polyphony_min"] = x.onset_polyphony_min;
    if (x.onset_polyphony_max) j["onset_polyphony_max"] = x.onset_polyphony_max;
    if (x.onset_density)       j["onset_density"] = x.onset_density;
    if (x.onset_density_min)   j["onset_density_min"] = x.onset_density_min;
    if (x.onset_density_max)   j["onset_density_max"] = x.onset_density_max;
    if (!x.duration_distribution.empty()) j["duration_distribution"] = x.duration_distribution;
    if (x.genre) j["genre"] = x.genre;
    if (x.note_density_level) j["note_density_level"] = x.note_density_level;
    if (x.pitch_class_count)  j["pitch_class_count"] = x.pitch_class_count;
    if (x.key_signature)      j["key_signature"] = x.key_signature;
    if (!x.pitch_classes.empty()) j["pitch_classes"] = x.pitch_classes;
    if (x.rest_percentage != 0) j["rest_percentage"] = x.rest_percentage;
    json acd; to_json(acd, x.attribute_control_distributions);
    if (!acd.empty()) j["attribute_control_distributions"] = acd;
}

// ─ PieceFeatures ─

inline void from_json(const json &j, PieceFeatures &x) {
    x.genre = j.value("genre", std::string{});
    x.danceability = j.value("danceability", 0);
}
inline void to_json(json &j, const PieceFeatures &x) {
    j = json{};
    if (!x.genre.empty())   j["genre"] = x.genre;
    if (x.danceability)     j["danceability"] = x.danceability;
}

// ─ GenreData ─

inline void from_json(const json &j, GenreData &x) {
    x.discogs = j.value("discogs", std::string{});
    x.lastfm  = j.value("lastfm", std::string{});
    x.tagtraum = j.value("tagtraum", std::string{});
}
inline void to_json(json &j, const GenreData &x) {
    j = json{};
    if (!x.discogs.empty())  j["discogs"]  = x.discogs;
    if (!x.lastfm.empty())   j["lastfm"]   = x.lastfm;
    if (!x.tagtraum.empty()) j["tagtraum"] = x.tagtraum;
}

// ─ MetadataLabels ─

inline void from_json(const json &j, MetadataLabels &x) {
    x.genre = j.value("genre", 0);
    x.valence_spotify     = j.value("valence_spotify", 0.0f);
    x.energy_spotify      = j.value("energy_spotify", 0.0f);
    x.danceability_spotify = j.value("danceability_spotify", 0.0f);
    if (j.contains("tension")) x.tension = j["tension"].get<std::vector<float>>();
    x.nomml = j.value("nomml", 0);
}
inline void to_json(json &j, const MetadataLabels &x) {
    j = json{};
    if (x.genre) j["genre"] = x.genre;
    if (x.valence_spotify != 0)     j["valence_spotify"] = x.valence_spotify;
    if (x.energy_spotify != 0)      j["energy_spotify"] = x.energy_spotify;
    if (x.danceability_spotify != 0) j["danceability_spotify"] = x.danceability_spotify;
    if (!x.tension.empty()) j["tension"] = x.tension;
    if (x.nomml) j["nomml"] = x.nomml;
}

// ─ ValidTrack ─

inline void from_json(const json &j, ValidTrack &x) {
    if (j.contains("tracks")) x.tracks = j["tracks"].get<std::vector<int>>();
}
inline void to_json(json &j, const ValidTrack &x) {
    j = json{};
    if (!x.tracks.empty()) j["tracks"] = x.tracks;
}

// ─ TempoChange ─

inline void from_json(const json &j, TempoChange &x) {
    x.tick = j.value("tick", 0);
    x.qpm  = j.value("qpm", 0.0f);
}
inline void to_json(json &j, const TempoChange &x) {
    j = json{{"tick", x.tick}, {"qpm", x.qpm}};
}

// ─ Event ─

inline void from_json(const json &j, Event &x) {
    x.time               = j.value("time", 0);
    x.velocity           = j.value("velocity", 0);
    x.pitch              = j.value("pitch", 0);
    x.internal_instrument = j.value("internal_instrument", 0);
    x.internal_track_type = j.value("internal_track_type", 0);
    x.internal_duration  = j.value("internal_duration", 0);
    x.delta              = j.value("delta", 0);
}
inline void to_json(json &j, const Event &x) {
    j = json{{"time", x.time}, {"velocity", x.velocity}, {"pitch", x.pitch}};
    if (x.internal_instrument) j["internal_instrument"] = x.internal_instrument;
    if (x.internal_track_type) j["internal_track_type"] = x.internal_track_type;
    if (x.internal_duration)   j["internal_duration"] = x.internal_duration;
    if (x.delta)               j["delta"] = x.delta;
}

// ─ Bar ─

inline void from_json(const json &j, Bar &x) {
    if (j.contains("events")) x.events = j["events"].get<std::vector<int>>();
    x.ts_numerator        = j.value("ts_numerator", 4);
    x.ts_denominator      = j.value("ts_denominator", 4);
    x.internal_beat_length = j.value("internal_beat_length", 0.0f);
    x.internal_has_notes  = j.value("internal_has_notes", false);
    if (j.contains("internal_feature"))  x.internal_feature  = j["internal_feature"].get<std::vector<ContinuousFeature>>();
    if (j.contains("internal_features")) x.internal_features = j["internal_features"].get<std::vector<BarFeatures>>();
    x.future = j.value("future", false);
}
inline void to_json(json &j, const Bar &x) {
    j = json{};
    if (!x.events.empty())        j["events"] = x.events;
    j["ts_numerator"]   = x.ts_numerator;
    j["ts_denominator"] = x.ts_denominator;
    if (x.internal_beat_length != 0) j["internal_beat_length"] = x.internal_beat_length;
    if (x.internal_has_notes)        j["internal_has_notes"] = x.internal_has_notes;
    if (!x.internal_feature.empty()) j["internal_feature"]  = x.internal_feature;
    if (!x.internal_features.empty()) j["internal_features"] = x.internal_features;
    if (x.future) j["future"] = x.future;
}

// ─ Track ─

inline void from_json(const json &j, Track &x) {
    if (j.contains("bars")) x.bars = j["bars"].get<std::vector<Bar>>();
    x.instrument = j.value("instrument", 0);
    x.track_type = j.value("track_type", (int)STANDARD_TRACK);
    if (j.contains("internal_train_types")) x.internal_train_types = j["internal_train_types"].get<std::vector<int>>();
    if (j.contains("internal_features"))   x.internal_features    = j["internal_features"].get<std::vector<TrackFeatures>>();
}
inline void to_json(json &j, const Track &x) {
    j = json{};
    j["bars"]       = x.bars;
    j["instrument"] = x.instrument;
    j["track_type"] = x.track_type;
    if (!x.internal_train_types.empty()) j["internal_train_types"] = x.internal_train_types;
    if (!x.internal_features.empty())   j["internal_features"]    = x.internal_features;
}

// ─ Piece ─

inline void from_json(const json &j, Piece &x) {
    if (j.contains("tracks")) x.tracks = j["tracks"].get<std::vector<Track>>();
    if (j.contains("events")) x.events = j["events"].get<std::vector<Event>>();
    x.resolution = j.value("resolution", 12);
    x.tempo      = j.value("tempo", 0);
    if (j.contains("internal_valid_segments"))    x.internal_valid_segments    = j["internal_valid_segments"].get<std::vector<int>>();
    if (j.contains("internal_valid_tracks"))      x.internal_valid_tracks      = j["internal_valid_tracks"].get<std::vector<uint32_t>>();
    x.internal_segment_length = j.value("internal_segment_length", 0);
    if (j.contains("internal_valid_tracks_v2"))  x.internal_valid_tracks_v2  = j["internal_valid_tracks_v2"].get<std::vector<ValidTrack>>();
    if (j.contains("internal_genre_data"))        x.internal_genre_data        = j["internal_genre_data"].get<std::vector<GenreData>>();
    if (j.contains("internal_metadata_labels"))   x.internal_metadata_labels   = j["internal_metadata_labels"].get<MetadataLabels>();
    if (j.contains("internal_features"))          x.internal_features          = j["internal_features"].get<std::vector<PieceFeatures>>();
    x.internal_ticks_per_quarter   = j.value("internal_ticks_per_quarter", 0);
    x.internal_has_time_signatures = j.value("internal_has_time_signatures", false);
    if (j.contains("tempo_changes")) x.tempo_changes = j["tempo_changes"].get<std::vector<TempoChange>>();
}
inline void to_json(json &j, const Piece &x) {
    j = json{};
    j["tracks"]     = x.tracks;
    j["events"]     = x.events;
    j["resolution"] = x.resolution;
    if (x.tempo) j["tempo"] = x.tempo;
    if (!x.internal_valid_segments.empty())   j["internal_valid_segments"]   = x.internal_valid_segments;
    if (!x.internal_valid_tracks.empty())     j["internal_valid_tracks"]     = x.internal_valid_tracks;
    if (x.internal_segment_length)            j["internal_segment_length"]   = x.internal_segment_length;
    if (!x.internal_valid_tracks_v2.empty())  j["internal_valid_tracks_v2"]  = x.internal_valid_tracks_v2;
    if (!x.internal_genre_data.empty())       j["internal_genre_data"]       = x.internal_genre_data;
    if (x.internal_ticks_per_quarter)         j["internal_ticks_per_quarter"] = x.internal_ticks_per_quarter;
    if (x.internal_has_time_signatures)       j["internal_has_time_signatures"] = x.internal_has_time_signatures;
    if (!x.tempo_changes.empty())             j["tempo_changes"] = x.tempo_changes;
}

// ─ StatusBar ─

inline void from_json(const json &j, StatusBar &x) {
    x.ts_numerator        = j.value("ts_numerator", 0);
    x.ts_denominator      = j.value("ts_denominator", 0);
    x.onset_density       = j.value("onset_density", 0);
    x.onset_polyphony_min = j.value("onset_polyphony_min", 0);
    x.onset_polyphony_max = j.value("onset_polyphony_max", 0);
    if (j.contains("pitch_class_set")) x.pitch_class_set = j["pitch_class_set"].get<std::vector<bool>>();
    x.tension      = j.value("tension", 0);
    x.future       = j.value("future", false);
    x.tension_drum = j.value("tension_drum", 0);
}
inline void to_json(json &j, const StatusBar &x) {
    j = json{};
    if (x.ts_numerator)        j["ts_numerator"]        = x.ts_numerator;
    if (x.ts_denominator)      j["ts_denominator"]      = x.ts_denominator;
    if (x.onset_density)       j["onset_density"]       = x.onset_density;
    if (x.onset_polyphony_min) j["onset_polyphony_min"] = x.onset_polyphony_min;
    if (x.onset_polyphony_max) j["onset_polyphony_max"] = x.onset_polyphony_max;
    if (!x.pitch_class_set.empty()) j["pitch_class_set"] = x.pitch_class_set;
    if (x.tension)             j["tension"]      = x.tension;
    if (x.future)              j["future"]       = x.future;
    if (x.tension_drum)        j["tension_drum"] = x.tension_drum;
}

// ─ StatusTrack ─

inline void from_json(const json &j, StatusTrack &x) {
    x.track_id   = j.value("track_id", 0);
    x.track_type = j.value("track_type", (int)STANDARD_TRACK);
    // instrument: accept string (GM name) or int
    if (j.contains("instrument")) {
        if (j["instrument"].is_string())
            x.instrument = gm_type_from_string(j["instrument"].get<std::string>());
        else
            x.instrument = j["instrument"].get<int>();
    }
    if (j.contains("selected_bars")) x.selected_bars = j["selected_bars"].get<std::vector<bool>>();
    x.autoregressive      = j.value("autoregressive", false);
    x.ignore              = j.value("ignore", false);
    x.density             = j.value("density", 0);
    x.min_polyphony_q     = j.value("min_polyphony_q", 0);
    x.max_polyphony_q     = j.value("max_polyphony_q", 0);
    x.min_note_duration_q = j.value("min_note_duration_q", 0);
    x.max_note_duration_q = j.value("max_note_duration_q", 0);
    x.onset_polyphony_min = j.value("onset_polyphony_min", 0);
    x.onset_polyphony_max = j.value("onset_polyphony_max", 0);
    x.onset_density       = j.value("onset_density", 0);
    x.onset_density_min   = j.value("onset_density_min", 0);
    x.onset_density_max   = j.value("onset_density_max", 0);
    x.min_pitch           = j.value("min_pitch", 0);
    x.max_pitch           = j.value("max_pitch", 0);
    x.genre               = j.value("genre", 0);
    x.silence_proportion_min = j.value("silence_proportion_min", 0);
    x.silence_proportion_max = j.value("silence_proportion_max", 0);
    x.note_density_level  = j.value("note_density_level", 0);
    x.contains_note_duration_thirty_second = j.value("contains_note_duration_thirty_second", 0);
    x.contains_note_duration_sixteenth     = j.value("contains_note_duration_sixteenth", 0);
    x.contains_note_duration_eighth        = j.value("contains_note_duration_eighth", 0);
    x.contains_note_duration_quarter       = j.value("contains_note_duration_quarter", 0);
    x.contains_note_duration_half          = j.value("contains_note_duration_half", 0);
    x.contains_note_duration_whole         = j.value("contains_note_duration_whole", 0);
    x.pitch_class_count   = j.value("pitch_class_count", 0);
    x.key_signature       = j.value("key_signature", 0);
    x.valence_spotify     = j.value("valence_spotify", 0);
    x.energy_spotify      = j.value("energy_spotify", 0);
    x.danceability_spotify = j.value("danceability_spotify", 0);
    x.danceability        = j.value("danceability", 0);
    x.wnbd_syncopation    = j.value("wnbd_syncopation", 0);
    x.repetition          = j.value("repetition", 0);
    x.suffix_autoregressive = j.value("suffix_autoregressive", false);
    x.polyphony_hard_limit  = j.value("polyphony_hard_limit", 0);
    x.temperature           = j.value("temperature", 0.0f);
    if (j.contains("internal_ts_numerators"))   x.internal_ts_numerators   = j["internal_ts_numerators"].get<std::vector<int>>();
    if (j.contains("internal_ts_denominators")) x.internal_ts_denominators = j["internal_ts_denominators"].get<std::vector<int>>();
    x.internal_genre = j.value("internal_genre", std::string{});
    if (j.contains("bars")) x.bars = j["bars"].get<std::vector<StatusBar>>();
}
inline void to_json(json &j, const StatusTrack &x) {
    j = json{};
    j["track_id"]   = x.track_id;
    j["track_type"] = x.track_type;
    j["instrument"] = x.instrument;
    if (!x.selected_bars.empty())  j["selected_bars"] = x.selected_bars;
    if (x.autoregressive)          j["autoregressive"] = x.autoregressive;
    if (x.ignore)                  j["ignore"] = x.ignore;
    if (x.density)                 j["density"] = x.density;
    if (x.min_polyphony_q)         j["min_polyphony_q"] = x.min_polyphony_q;
    if (x.max_polyphony_q)         j["max_polyphony_q"] = x.max_polyphony_q;
    if (x.min_note_duration_q)     j["min_note_duration_q"] = x.min_note_duration_q;
    if (x.max_note_duration_q)     j["max_note_duration_q"] = x.max_note_duration_q;
    if (x.onset_polyphony_min)     j["onset_polyphony_min"] = x.onset_polyphony_min;
    if (x.onset_polyphony_max)     j["onset_polyphony_max"] = x.onset_polyphony_max;
    if (x.onset_density)           j["onset_density"] = x.onset_density;
    if (x.onset_density_min)       j["onset_density_min"] = x.onset_density_min;
    if (x.onset_density_max)       j["onset_density_max"] = x.onset_density_max;
    if (x.min_pitch)               j["min_pitch"] = x.min_pitch;
    if (x.max_pitch)               j["max_pitch"] = x.max_pitch;
    if (x.genre)                   j["genre"] = x.genre;
    if (x.polyphony_hard_limit)    j["polyphony_hard_limit"] = x.polyphony_hard_limit;
    if (x.temperature != 0)        j["temperature"] = x.temperature;
    if (x.suffix_autoregressive)   j["suffix_autoregressive"] = x.suffix_autoregressive;
    if (!x.bars.empty())           j["bars"] = x.bars;
    if (!x.internal_ts_numerators.empty())   j["internal_ts_numerators"]   = x.internal_ts_numerators;
    if (!x.internal_ts_denominators.empty()) j["internal_ts_denominators"] = x.internal_ts_denominators;
    if (!x.internal_genre.empty()) j["internal_genre"] = x.internal_genre;
}

// ─ Status ─

inline void from_json(const json &j, Status &x) {
    if (j.contains("tracks"))  x.tracks  = j["tracks"].get<std::vector<StatusTrack>>();
    x.decode_final    = j.value("decode_final", false);
    x.full_resolution = j.value("full_resolution", false);
}
inline void to_json(json &j, const Status &x) {
    j = json{{"tracks", x.tracks}};
    if (x.decode_final)    j["decode_final"] = x.decode_final;
    if (x.full_resolution) j["full_resolution"] = x.full_resolution;
}

// ─ HyperParam ─

inline void from_json(const json &j, HyperParam &x) {
    x.tracks_per_step = j.value("tracks_per_step", 1);
    x.bars_per_step   = j.value("bars_per_step", 1);
    x.model_dim       = j.value("model_dim", 4);
    x.percentage      = j.value("percentage", 100);
    x.batch_size      = j.value("batch_size", 1);
    x.temperature     = j.value("temperature", 1.0f);
    x.use_per_track_temperature = j.value("use_per_track_temperature", false);
    x.max_steps       = j.value("max_steps", 0);
    x.polyphony_hard_limit = j.value("polyphony_hard_limit", 0);
    x.shuffle         = j.value("shuffle", false);
    x.verbose         = j.value("verbose", false);
    x.ckpt            = j.value("ckpt", std::string{});
    x.mask_top_k      = j.value("mask_top_k", 0.0f);
    x.sampling_seed   = j.value("sampling_seed", -1);
    x.internal_skip_preprocess   = j.value("internal_skip_preprocess", false);
    x.internal_disable_masking   = j.value("internal_disable_masking", false);
}
inline void to_json(json &j, const HyperParam &x) {
    j = json{};
    j["tracks_per_step"] = x.tracks_per_step;
    j["bars_per_step"]   = x.bars_per_step;
    j["model_dim"]       = x.model_dim;
    j["percentage"]      = x.percentage;
    j["batch_size"]      = x.batch_size;
    j["temperature"]     = x.temperature;
    if (x.use_per_track_temperature) j["use_per_track_temperature"] = x.use_per_track_temperature;
    if (x.max_steps)     j["max_steps"] = x.max_steps;
    if (x.polyphony_hard_limit) j["polyphony_hard_limit"] = x.polyphony_hard_limit;
    if (x.shuffle)       j["shuffle"] = x.shuffle;
    if (x.verbose)       j["verbose"] = x.verbose;
    if (!x.ckpt.empty()) j["ckpt"]    = x.ckpt;
    if (x.mask_top_k != 0) j["mask_top_k"] = x.mask_top_k;
    if (x.sampling_seed != -1) j["sampling_seed"] = x.sampling_seed;
    if (x.internal_skip_preprocess)  j["internal_skip_preprocess"] = x.internal_skip_preprocess;
    if (x.internal_disable_masking)  j["internal_disable_masking"] = x.internal_disable_masking;
}

// ── Convenience: string ↔ struct ─────────────────────────────────────────────

inline Piece   piece_from_string(const std::string &s)  { return json::parse(s).get<Piece>(); }
inline Status  status_from_string(const std::string &s) { return json::parse(s).get<Status>(); }
inline HyperParam param_from_string(const std::string &s) { return json::parse(s).get<HyperParam>(); }

inline std::string piece_to_string(const Piece &x)     { return json(x).dump(); }
inline std::string status_to_string(const Status &x)   { return json(x).dump(); }
inline std::string param_to_string(const HyperParam &x){ return json(x).dump(); }

} // namespace midi
