# MIDI-GPT Refactor Implementation Plan

## Repository layout

```
midigpt/
├── src/
│   ├── cpp/
│   │   ├── core/
│   │   │   ├── types.h               # All enums: TokenType, TrackType, BooleanEnum, …
│   │   │   └── score.h               # Note, Bar, Track, Score structs
│   │   ├── io/
│   │   │   ├── midi_reader.h / .cpp  # symusic → Score
│   │   │   └── midi_writer.h / .cpp  # Score → symusic → file
│   │   ├── tokenizer/
│   │   │   ├── encoder_config.h / .cpp  # EncoderConfig + JSON load/save
│   │   │   ├── vocabulary.h  / .cpp     # Vocabulary
│   │   │   ├── encoder.h     / .cpp     # Score + attrs → tokens
│   │   │   └── decoder.h     / .cpp     # tokens → Score
│   │   ├── masking/
│   │   │   ├── constraint.h             # Abstract base Constraint
│   │   │   ├── constraint_graph.h/.cpp  # AND-composition of constraints
│   │   │   ├── grammar_constraint.h     # Token sequence validity (FSM)
│   │   │   ├── polyphony_constraint.h   # Max simultaneous notes
│   │   │   └── density_constraint.h     # Notes-per-bar bounds
│   │   ├── sampling/
│   │   │   ├── selection_mask.h         # SelectionMask struct
│   │   │   ├── generation_step.h        # GenerationStep struct
│   │   │   ├── step_planner.h / .cpp    # StepPlanner
│   │   │   └── session_state.h / .cpp   # SessionState
│   │   └── bindings/
│   │       └── lib.cpp                  # pybind11 — pure translation, zero logic
│   └── python/
│       └── midigpt/
│           ├── __init__.py              # Public API surface
│           ├── _types.py                # Score, Track, Bar, Note dataclasses
│           ├── _converters.py           # to_cpp() / from_cpp() — internal only
│           ├── attributes/
│           │   ├── __init__.py
│           │   ├── base.py              # BaseAttribute, AttributeAnalyzer
│           │   ├── density.py
│           │   ├── polyphony.py
│           │   ├── pitch_range.py
│           │   ├── key_signature.py
│           │   ├── note_duration.py
│           │   ├── tension.py
│           │   ├── silence.py
│           │   └── pitch_class_set.py
│           ├── augmentation/
│           │   ├── __init__.py
│           │   ├── base.py              # BaseTransform, AugmentationPipeline
│           │   ├── transpose.py
│           │   ├── velocity.py
│           │   ├── track_permutation.py
│           │   ├── bar_window.py
│           │   └── instrument_swap.py
│           ├── tokenizer/
│           │   ├── __init__.py
│           │   ├── tokenizer.py         # Tokenizer — Python orchestrator
│           │   └── checkpoint.py        # load_checkpoint(), CheckpointBundle
│           ├── inference/               # [inference] extra
│           │   ├── __init__.py
│           │   ├── config.py            # SamplingConfig, TrackPrompt, GenerationRequest
│           │   ├── engine.py            # InferenceEngine
│           │   ├── session.py           # SamplingSession
│           │   └── realtime_session.py  # RealtimeSession
│           ├── server/                  # [osc] extra
│           │   ├── __init__.py
│           │   ├── osc_server.py        # MidiGPTServer — dispatcher
│           │   ├── protocol.py          # OSC message parsing / formatting
│           │   └── state.py             # RealtimeState — bar/note/track machine
│           └── training/                # [train] extra
│               ├── __init__.py
│               ├── dataset.py           # DatasetBuilder, MidiGPTDataset
│               ├── collator.py          # DataCollator for CLM
│               └── trainer.py           # HF Trainer wrapper + helpers
├── tests/
│   ├── cpp/                             # doctest — header-only, zero setup
│   │   ├── test_score.cpp
│   │   ├── test_vocabulary.cpp
│   │   ├── test_encoder.cpp
│   │   ├── test_decoder.cpp
│   │   ├── test_roundtrip.cpp
│   │   ├── test_constraint_graph.cpp
│   │   ├── test_step_planner.cpp
│   │   └── test_session_state.cpp
│   └── python/                          # pytest
│       ├── conftest.py
│       ├── test_types.py
│       ├── test_converters.py
│       ├── test_attributes.py
│       ├── test_augmentation.py
│       ├── test_tokenizer.py
│       ├── test_inference.py
│       ├── test_training.py
│       └── test_server.py
├── include/
│   └── nlohmann/
│       └── json.hpp                     # vendored single header
├── cmake/
│   ├── dependencies.cmake
│   └── compiler_flags.cmake
├── CMakeLists.txt
└── pyproject.toml
```

---

## C++ modules

### `src/cpp/core/types.h`

All enums. Zero dependencies.

```cpp
namespace midigpt {

enum class TokenType {
    PieceStart, Track, TrackEnd, Bar, BarEnd,
    Instrument, NoteOnset, NoteDuration, TimeAbsolutePos,
    TimeSig, VelocityLevel, DeltaDirection, Delta,
    FillInStart, FillInEnd, FillInPlaceholder, MaskBar,
    NumBars, PieceEnd,
    // Attribute tokens
    NoteDensity, OnsetPolyphony, PitchRange, KeySignature,
    NoteDurationDist, Tension, SilenceProportion, PitchClassSet,
    // ... full list mirrors current TOKEN_TYPE
};

enum class TrackType { Melodic = 0, Drum = 1 };

// Used by attribute constraints: ANY means unconstrained
enum class BooleanEnum { Any = 0, False = 1, True = 2 };

} // namespace midigpt
```

---

### `src/cpp/core/score.h`

```cpp
namespace midigpt {

struct Note {
    int pitch;
    int velocity;
    int onset_ticks;
    int duration_ticks;
    int delta = 0;          // microtiming offset from onset
};

struct Bar {
    std::vector<int> note_indices;  // indices into Score::notes pool
    int  ts_numerator   = 4;
    int  ts_denominator = 4;
    int  beat_length    = 0;        // ticks per bar, filled during decode
    bool has_notes      = false;
    bool future         = false;    // true → encode as MASK_BAR
};

struct Track {
    std::vector<Bar> bars;
    int       instrument = 0;
    TrackType type       = TrackType::Melodic;
};

struct Score {
    std::vector<Track> tracks;
    std::vector<Note>  notes;   // global pool; Bars index into this
    int resolution = 480;       // ticks per quarter note
    int tempo      = 500000;    // microseconds per beat
};

} // namespace midigpt
```

---

### `src/cpp/io/`

**`midi_reader.h`**
```cpp
namespace midigpt::io {

class MidiReader {
public:
    Score read(const std::string& path) const;
    Score read_bytes(const std::vector<uint8_t>& bytes) const;
private:
    // converts symusic Score → midigpt::Score
    Score from_symusic(const symusic::Score<symusic::Tick>& s) const;
};

} // namespace midigpt::io
```

**`midi_writer.h`**
```cpp
namespace midigpt::io {

class MidiWriter {
public:
    void write(const Score& score, const std::string& path) const;
    std::vector<uint8_t> write_bytes(const Score& score) const;
private:
    symusic::Score<symusic::Tick> to_symusic(const Score& score) const;
};

} // namespace midigpt::io
```

symusic is the **only** external C++ dependency. Everything above this layer is zero-dep.

---

### `src/cpp/tokenizer/encoder_config.h`

```cpp
namespace midigpt::tokenizer {

struct TokenDomain {
    TokenType type;
    int       domain_size;
};

struct EncoderConfig {
    int  resolution        = 480;
    int  model_dim         = 4;     // bars per generation step
    bool use_microtiming   = false;
    int  decode_resolution = 480;   // target resolution after microtiming resample

    // Ordered list of active token types and their domain sizes.
    // Determines vocabulary layout: offset[i] = sum of domain_sizes[0..i-1]
    std::vector<TokenDomain> token_domains;

    static EncoderConfig from_json(const std::string& json_str); // uses nlohmann
    std::string          to_json()                          const;
};

} // namespace midigpt::tokenizer
```

---

### `src/cpp/tokenizer/vocabulary.h`

```cpp
namespace midigpt::tokenizer {

class Vocabulary {
public:
    explicit Vocabulary(const EncoderConfig& config);

    // Encoding
    int encode(TokenType type, int value)               const;
    int encode(TokenType type, std::tuple<int,int> ts)  const; // time signatures

    // Decoding
    std::pair<TokenType, int> decode(int token)         const;

    // Queries
    int  size()                                         const;
    bool has(TokenType type)                            const;
    int  domain_size(TokenType type)                    const;
    int  offset(TokenType type)                         const; // start index in vocab
    bool is_type(int token, TokenType type)             const;
    TokenType get_type(int token)                       const;

    // Range [start, end) of token integers for a given type — used by masking
    std::pair<int,int> range(TokenType type)            const;

    const EncoderConfig& config() const { return config_; }

private:
    EncoderConfig                     config_;
    std::unordered_map<TokenType,int> offsets_;
    int                               vocab_size_;
};

} // namespace midigpt::tokenizer
```

---

### `src/cpp/tokenizer/encoder.h`

```cpp
namespace midigpt::tokenizer {

// Attribute values pre-computed by Python AttributeAnalyzer.
// Maps TokenType → quantized integer value [0, domain_size).
using TokenAttrs = std::unordered_map<TokenType, int>;

struct TrackAttrs { TokenAttrs values; };
struct BarAttrs   { TokenAttrs values; };

// Controls which bars are context vs generation targets (for context encoding)
struct EncodingMask {
    std::vector<std::vector<bool>> selected;      // [track][bar] — true = generate
    std::set<std::pair<int,int>>   infill_targets; // (track, bar) for multi-fill
    int partial_encode_track = -1;  // suffix-autoregressive: encode N bars of track K
    int partial_encode_bars  = -1;
};

class Encoder {
public:
    Encoder(const Vocabulary& vocab, const EncoderConfig& config);

    // Full encode — for training, all bars included
    std::vector<int> encode(
        const Score&                              score,
        const std::vector<TrackAttrs>&            track_attrs,
        const std::vector<std::vector<BarAttrs>>& bar_attrs
    ) const;

    // Context encode — for generation, respects selection mask
    std::vector<int> encode_context(
        const Score&                              score,
        const std::vector<TrackAttrs>&            track_attrs,
        const std::vector<std::vector<BarAttrs>>& bar_attrs,
        const EncodingMask&                       mask
    ) const;

private:
    void encode_notes(const Score&, int track, int bar, std::vector<int>& out)               const;
    void encode_bar  (const Score&, int track, int bar,
                      const BarAttrs&, const EncodingMask&, std::vector<int>& out)           const;
    void encode_track(const Score&, int track,
                      const TrackAttrs&, const std::vector<BarAttrs>&,
                      const EncodingMask&, std::vector<int>& out)                            const;

    const Vocabulary&    vocab_;
    const EncoderConfig& config_;
};

} // namespace midigpt::tokenizer
```

---

### `src/cpp/tokenizer/decoder.h`

```cpp
namespace midigpt::tokenizer {

class Decoder {
public:
    Decoder(const Vocabulary& vocab, const EncoderConfig& config);
    Score decode(const std::vector<int>& tokens) const;

private:
    // State machine helpers — mirrors current decode_track logic, now inside a class
    void handle_track_token  (int token, Score&, Track*&, int& track_count)         const;
    void handle_bar_token    (int token, Score&, Track*, Bar*&, int beat_length,
                              std::set<int>& offset_remain)                          const;
    void handle_note_onset   (int token, Score&, Bar*, int time, int vel, int delta) const;
    void handle_note_duration(int token, Score&, Bar*, int last_onset_token,
                              int time, int vel, int delta, int beat_length,
                              std::set<int>& offset_remain)                          const;

    const Vocabulary&    vocab_;
    const EncoderConfig& config_;
};

} // namespace midigpt::tokenizer
```

---

### `src/cpp/masking/`

**`constraint.h`** — abstract base:
```cpp
namespace midigpt::masking {

class Constraint {
public:
    virtual ~Constraint() = default;
    // true = token is valid at this position
    virtual std::vector<bool> compute(const std::vector<int>& generated_tokens,
                                      const Vocabulary&        vocab) const = 0;
};

} // namespace midigpt::masking
```

**`constraint_graph.h`** — AND-composition:
```cpp
namespace midigpt::masking {

class ConstraintGraph {
public:
    void add(std::unique_ptr<Constraint> c);

    // Intersects all constraint masks — result is valid where ALL are true
    std::vector<bool> compute_mask(const std::vector<int>& generated_tokens,
                                   const Vocabulary&        vocab) const;
private:
    std::vector<std::unique_ptr<Constraint>> constraints_;
};

} // namespace midigpt::masking
```

**Concrete constraints** (each a single `.h` file):

| File | What it enforces |
|---|---|
| `grammar_constraint.h` | Token type sequencing FSM (TRACK before BAR, NOTE_ONSET before NOTE_DURATION, etc.) |
| `polyphony_constraint.h` | Counts active simultaneous notes; masks NOTE_ONSET when limit reached |
| `density_constraint.h` | Tracks note count per bar; masks NOTE_ONSET above max, un-masks BAR_END below min |
| `attribute_value_constraint.h` | For a given TokenType, masks all values except the one specified in the request |

---

### `src/cpp/sampling/`

**`selection_mask.h`**:
```cpp
namespace midigpt::sampling {

struct SelectionMask {
    std::vector<std::vector<bool>> selected;   // [track][bar] — true = generate here
    std::vector<bool> autoregressive;          // per track
    std::vector<bool> ignore;                  // per track — excluded from context
};

} // namespace midigpt::sampling
```

**`generation_step.h`**:
```cpp
namespace midigpt::sampling {

struct GenerationStep {
    int start_bar;
    int end_bar;
    std::vector<int>                          track_indices;    // all tracks in window
    std::set<std::pair<int,int>>              bars_to_generate; // (track, bar)
    std::vector<std::tuple<int,int,int,int>>  bar_mapping;      // src_t,src_b,dst_t,dst_b
};

} // namespace midigpt::sampling
```

**`step_planner.h`**:
```cpp
namespace midigpt::sampling {

class StepPlanner {
public:
    StepPlanner(const SelectionMask& mask, const EncoderConfig& config);
    std::vector<GenerationStep> plan() const;

private:
    // Replicates current multi_step find_steps logic — now a proper class
    void find_autoregressive_steps(std::vector<GenerationStep>&, ...) const;
    void find_infill_steps        (std::vector<GenerationStep>&, ...) const;

    SelectionMask mask_;
    EncoderConfig config_;
};

} // namespace midigpt::sampling
```

**`session_state.h`**:
```cpp
namespace midigpt::sampling {

class SessionState {
public:
    SessionState(
        Score                                     context,
        const GenerationStep&                     step,
        const Vocabulary&                         vocab,
        const ConstraintGraph&                    constraints,
        const Encoder&                            encoder,
        const Decoder&                            decoder,
        const std::vector<TrackAttrs>&            track_attrs,
        const std::vector<std::vector<BarAttrs>>& bar_attrs
    );

    bool              complete()       const; // all bars_to_generate are done
    std::vector<int>  context_tokens() const; // full context for model forward
    std::vector<bool> logit_mask()     const; // from ConstraintGraph — valid next tokens
    void              advance(int token);     // append token, update internal state
    Score             result()         const; // decode + apply generated bars into context

private:
    Score              context_;
    GenerationStep     step_;
    const Vocabulary&  vocab_;
    ConstraintGraph    constraints_;      // owns a copy — constraints are step-local
    const Encoder&     encoder_;
    const Decoder&     decoder_;
    std::vector<int>   generated_;       // tokens produced so far in this step
    std::vector<int>   context_cache_;   // pre-encoded context (immutable during step)
};

} // namespace midigpt::sampling
```

---

### `src/cpp/bindings/lib.cpp`

Pure translation layer. No business logic. Exposes all C++ types to Python with appropriate type mappings:

- `std::vector<bool>` → `numpy` bool array for masks (avoids Python list overhead)
- `std::vector<int>` → Python `list[int]`
- C++ struct fields → Python attributes with getters/setters
- Factory methods (e.g. `EncoderConfig::from_json`) → `@staticmethod`

```cpp
PYBIND11_MODULE(_core, m) {

    // enums
    py::enum_<TokenType>(m, "TokenType") /* .value(...) for all */ ;
    py::enum_<TrackType>(m, "TrackType").value("Melodic", TrackType::Melodic)
                                        .value("Drum",    TrackType::Drum);

    // core structs
    py::class_<Note>(m, "Note")
        .def(py::init<int,int,int,int,int>(), ...)
        .def_readwrite("pitch", &Note::pitch) /* ... */ ;

    py::class_<Bar>(m, "Bar")   /* ... */ ;
    py::class_<Track>(m, "Track") /* ... */ ;
    py::class_<Score>(m, "Score") /* ... */ ;

    // io
    py::class_<MidiReader>(m, "MidiReader")
        .def(py::init<>())
        .def("read",       &MidiReader::read)
        .def("read_bytes", &MidiReader::read_bytes);

    py::class_<MidiWriter>(m, "MidiWriter")
        .def(py::init<>())
        .def("write",       &MidiWriter::write)
        .def("write_bytes", &MidiWriter::write_bytes);

    // tokenizer
    py::class_<EncoderConfig>(m, "EncoderConfig")
        .def_static("from_json", &EncoderConfig::from_json)
        .def("to_json",          &EncoderConfig::to_json)
        .def_readwrite("resolution",      &EncoderConfig::resolution)
        .def_readwrite("model_dim",       &EncoderConfig::model_dim)
        .def_readwrite("use_microtiming", &EncoderConfig::use_microtiming);

    py::class_<Vocabulary>(m, "Vocabulary")
        .def(py::init<const EncoderConfig&>())
        .def("encode",      py::overload_cast<TokenType,int>(&Vocabulary::encode, py::const_))
        .def("decode",      &Vocabulary::decode)
        .def("size",        &Vocabulary::size)
        .def("has",         &Vocabulary::has)
        .def("domain_size", &Vocabulary::domain_size)
        .def("range",       &Vocabulary::range)
        .def("config",      &Vocabulary::config, py::return_value_policy::reference);

    py::class_<TrackAttrs>(m, "TrackAttrs").def(py::init<std::unordered_map<TokenType,int>>());
    py::class_<BarAttrs>(m, "BarAttrs")   .def(py::init<std::unordered_map<TokenType,int>>());
    py::class_<EncodingMask>(m, "EncodingMask") /* ... */ ;

    py::class_<Encoder>(m, "Encoder")
        .def(py::init<const Vocabulary&, const EncoderConfig&>())
        .def("encode",         &Encoder::encode)
        .def("encode_context", &Encoder::encode_context);

    py::class_<Decoder>(m, "Decoder")
        .def(py::init<const Vocabulary&, const EncoderConfig&>())
        .def("decode", &Decoder::decode);

    // masking
    py::class_<ConstraintGraph>(m, "ConstraintGraph")
        .def(py::init<>())
        .def("add_grammar_constraint",        ...)
        .def("add_polyphony_constraint",      ...)
        .def("add_density_constraint",        ...)
        .def("add_attribute_value_constraint", ...)
        .def("compute_mask", &ConstraintGraph::compute_mask);

    // sampling
    py::class_<SelectionMask>(m, "SelectionMask")
        .def(py::init<std::vector<std::vector<bool>>, std::vector<bool>, std::vector<bool>>());

    py::class_<GenerationStep>(m, "GenerationStep") /* ... */ ;

    py::class_<StepPlanner>(m, "StepPlanner")
        .def(py::init<const SelectionMask&, const EncoderConfig&>())
        .def("plan", &StepPlanner::plan);

    py::class_<SessionState>(m, "SessionState")
        .def(py::init< Score, const GenerationStep&, const Vocabulary&,
                       const ConstraintGraph&, const Encoder&, const Decoder&,
                       const std::vector<TrackAttrs>&,
                       const std::vector<std::vector<BarAttrs>>& >())
        .def("complete",       &SessionState::complete)
        .def("context_tokens", &SessionState::context_tokens)
        .def("logit_mask",     &SessionState::logit_mask)  // returns numpy bool array
        .def("advance",        &SessionState::advance)
        .def("result",         &SessionState::result);
}
```

---

## Python modules

### `midigpt/_types.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Note:
    pitch:          int
    velocity:       int
    onset_ticks:    int
    duration_ticks: int
    delta:          int = 0

@dataclass
class Bar:
    notes:          list[Note] = field(default_factory=list)
    ts_numerator:   int  = 4
    ts_denominator: int  = 4
    future:         bool = False

@dataclass
class Track:
    bars:       list[Bar] = field(default_factory=list)
    instrument: int       = 0
    track_type: str       = "melodic"   # "melodic" | "drum"

@dataclass
class Score:
    tracks:     list[Track] = field(default_factory=list)
    resolution: int         = 480
    tempo:      int         = 500000

    @classmethod
    def from_midi(cls, path: str) -> Score:
        from midigpt._converters import from_cpp
        from midigpt._core import MidiReader
        return from_cpp(MidiReader().read(path))

    def to_midi(self, path: str) -> None:
        from midigpt._converters import to_cpp
        from midigpt._core import MidiWriter
        MidiWriter().write(to_cpp(self), path)

    @classmethod
    def from_dict(cls, d: dict) -> Score: ...

    def to_dict(self) -> dict: ...
```

---

### `midigpt/_converters.py`

Internal. Converts between Python `Score` dataclass and C++ `_core.Score`. The key detail: Python Score stores notes directly on each Bar; C++ Score uses a global note pool indexed by Bar. The converter handles this mapping transparently.

```python
import midigpt._core as _core
from midigpt._types import Score, Track, Bar, Note

def to_cpp(score: Score) -> _core.Score:
    cpp = _core.Score()
    cpp.resolution = score.resolution
    cpp.tempo      = score.tempo
    note_pool: list[Note] = []
    for track in score.tracks:
        cpp_track = _core.Track()
        cpp_track.instrument = track.instrument
        cpp_track.type = (_core.TrackType.Drum
                          if track.track_type == "drum"
                          else _core.TrackType.Melodic)
        for bar in track.bars:
            cpp_bar = _core.Bar()
            cpp_bar.ts_numerator   = bar.ts_numerator
            cpp_bar.ts_denominator = bar.ts_denominator
            cpp_bar.future         = bar.future
            for note in bar.notes:
                cpp_bar.note_indices.append(len(note_pool))
                note_pool.append(note)
            cpp_track.bars.append(cpp_bar)
        cpp.tracks.append(cpp_track)
    for note in note_pool:
        cpp_note = _core.Note()
        cpp_note.pitch          = note.pitch
        cpp_note.velocity       = note.velocity
        cpp_note.onset_ticks    = note.onset_ticks
        cpp_note.duration_ticks = note.duration_ticks
        cpp_note.delta          = note.delta
        cpp.notes.append(cpp_note)
    return cpp

def from_cpp(cpp: _core.Score) -> Score:
    pool = [Note(n.pitch, n.velocity, n.onset_ticks, n.duration_ticks, n.delta)
            for n in cpp.notes]
    tracks = []
    for ct in cpp.tracks:
        bars = []
        for cb in ct.bars:
            bars.append(Bar(
                notes          = [pool[i] for i in cb.note_indices],
                ts_numerator   = cb.ts_numerator,
                ts_denominator = cb.ts_denominator,
                future         = cb.future,
            ))
        tracks.append(Track(
            bars       = bars,
            instrument = ct.instrument,
            track_type = "drum" if ct.type == _core.TrackType.Drum else "melodic",
        ))
    return Score(tracks=tracks, resolution=cpp.resolution, tempo=cpp.tempo)
```

---

### `midigpt/attributes/base.py`

```python
from abc import ABC, abstractmethod
from midigpt._types import Score

class BaseAttribute(ABC):
    name:       str   # e.g. "note_density"
    token_type: str   # e.g. "NoteDensity" — matches TokenType enum value name
    level:      str   # "track" | "bar"
    track_type: str   # "melodic" | "drum" | "both"

    @abstractmethod
    def compute(self, score: Score, track_idx: int,
                bar_idx: int | None = None) -> float | int: ...

    @abstractmethod
    def quantize(self, value: float | int) -> int: ...

    def dequantize(self, quantized: int) -> float | int:
        raise NotImplementedError


class AttributeAnalyzer:
    def __init__(self, attributes: list[BaseAttribute]):
        self._attrs = {a.name: a for a in attributes}

    def compute_track_tokens(self, score: Score, track_idx: int) -> dict[str, int]:
        """Returns {token_type_name: quantized_value} for track-level attrs."""
        result = {}
        track  = score.tracks[track_idx]
        is_drum = track.track_type == "drum"
        for attr in self._attrs.values():
            if attr.level != "track":
                continue
            if attr.track_type == "melodic" and is_drum:
                continue
            if attr.track_type == "drum" and not is_drum:
                continue
            raw = attr.compute(score, track_idx)
            result[attr.token_type] = attr.quantize(raw)
        return result

    def compute_bar_tokens(self, score: Score,
                           track_idx: int, bar_idx: int) -> dict[str, int]:
        """Returns {token_type_name: quantized_value} for bar-level attrs."""
        return {
            attr.token_type: attr.quantize(attr.compute(score, track_idx, bar_idx))
            for attr in self._attrs.values()
            if attr.level == "bar"
        }

    def compute_all(self, score: Score) -> list[dict[str, float | int]]:
        """Raw (unquantized) values per track — for evaluation."""
        return [
            {name: a.compute(score, i)
             for name, a in self._attrs.items()
             if a.level == "track"}
            for i in range(len(score.tracks))
        ]

    def evaluate(self, requested: dict[str, int],
                 realized_score: Score, track_idx: int) -> dict[str, float]:
        """Per-attribute match score in [0.0, 1.0]."""
        result = {}
        for name, req_q in requested.items():
            attr = self._attrs.get(name)
            if attr is None:
                continue
            raw      = attr.compute(realized_score, track_idx)
            real_q   = attr.quantize(raw)
            result[name] = 1.0 if real_q == req_q else 0.0
        return result

    @staticmethod
    def default() -> "AttributeAnalyzer":
        from midigpt.attributes import (
            NoteDensity, OnsetPolyphony, PitchRange, KeySignature,
            NoteDurationDistribution, Tension, SilenceProportion,
            BarLevelPitchClassSet,
        )
        return AttributeAnalyzer([
            NoteDensity(), OnsetPolyphony(), PitchRange(), KeySignature(),
            NoteDurationDistribution(), Tension(), SilenceProportion(),
            BarLevelPitchClassSet(),
        ])
```

Each concrete attribute is a small class in its own file implementing `compute()` and `quantize()`. No further dependencies — only reads `Score`.

---

### `midigpt/augmentation/base.py`

```python
from abc import ABC, abstractmethod
from midigpt._types import Score

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, score: Score) -> Score: ...

class AugmentationPipeline:
    def __init__(self, transforms: list[BaseTransform]):
        self._transforms = transforms

    def __call__(self, score: Score) -> Score:
        for t in self._transforms:
            score = t(score)
        return score

    @staticmethod
    def default_training() -> "AugmentationPipeline":
        from midigpt.augmentation import (
            Transpose, VelocityScale, TrackPermutation, BarWindow
        )
        return AugmentationPipeline([
            Transpose(range(-6, 7)),     # ±6 semitones, drums excluded
            VelocityScale((0.8, 1.2)),   # ±20%
            TrackPermutation(),          # shuffle track order
            BarWindow(num_bars=16),      # random 16-bar window
        ])
```

Concrete transforms — each `__call__` does `deepcopy` then mutates and returns:

| File | Class | Behaviour |
|---|---|---|
| `transpose.py` | `Transpose(semitones: int \| range)` | Shift pitch on melodic tracks only |
| `velocity.py` | `VelocityScale(factor: float \| tuple)` | Multiply velocity, clamp [1,127] |
| `track_permutation.py` | `TrackPermutation()` | `random.shuffle(score.tracks)` |
| `bar_window.py` | `BarWindow(num_bars: int)` | Slice random contiguous N bars |
| `instrument_swap.py` | `InstrumentSwap(mapping: dict[int, list[int]])` | Replace instrument with random choice from mapping |

---

### `midigpt/tokenizer/checkpoint.py`

```python
import pathlib
from dataclasses import dataclass
import midigpt._core as _core

@dataclass
class CheckpointBundle:
    model_path:     str
    encoder_config: _core.EncoderConfig

def load_checkpoint(path: str) -> CheckpointBundle:
    p = pathlib.Path(path)
    if not p.is_dir():
        raise ValueError(f"Checkpoint must be a directory: {path}")
    config_path = p / "config.json"
    model_path  = p / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing in: {path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt missing in: {path}")
    return CheckpointBundle(
        model_path     = str(model_path),
        encoder_config = _core.EncoderConfig.from_json(config_path.read_text()),
    )
```

Checkpoint directory format:
```
my_model/
├── model.pt       # TorchScript — torch.jit.load()
└── config.json    # EncoderConfig — bundled, version-locked to model
```

---

### `midigpt/tokenizer/tokenizer.py`

```python
import midigpt._core as _core
from midigpt._types import Score
from midigpt._converters import to_cpp, from_cpp
from midigpt.attributes.base import AttributeAnalyzer

class Tokenizer:
    """
    Stateless orchestrator: computes attributes in Python,
    delegates token sequence assembly to C++.
    Safe to use across DataLoader workers.
    """

    def __init__(self, encoder_config: _core.EncoderConfig,
                 analyzer: AttributeAnalyzer | None = None):
        self._vocab    = _core.Vocabulary(encoder_config)
        self._encoder  = _core.Encoder(self._vocab, encoder_config)
        self._decoder  = _core.Decoder(self._vocab, encoder_config)
        self._analyzer = analyzer or AttributeAnalyzer.default()

    def encode(self, score: Score) -> list[int]:
        ta, ba = self._compute_attrs(score)
        return self._encoder.encode(to_cpp(score), ta, ba)

    def encode_context(self, score: Score,
                       mask: _core.EncodingMask) -> list[int]:
        ta, ba = self._compute_attrs(score)
        return self._encoder.encode_context(to_cpp(score), ta, ba, mask)

    def decode(self, tokens: list[int]) -> Score:
        return from_cpp(self._decoder.decode(tokens))

    def vocab_size(self) -> int:
        return self._vocab.size()

    def _compute_attrs(self, score: Score):
        track_attrs, bar_attrs = [], []
        for i, track in enumerate(score.tracks):
            ta = self._analyzer.compute_track_tokens(score, i)
            track_attrs.append(_core.TrackAttrs(ta))
            bar_attrs.append([
                _core.BarAttrs(self._analyzer.compute_bar_tokens(score, i, j))
                for j in range(len(track.bars))
            ])
        return track_attrs, bar_attrs

    @classmethod
    def from_checkpoint(cls, path: str,
                        analyzer: AttributeAnalyzer | None = None) -> "Tokenizer":
        from midigpt.tokenizer.checkpoint import load_checkpoint
        bundle = load_checkpoint(path)
        return cls(bundle.encoder_config, analyzer)
```

---

### `midigpt/inference/config.py`

```python
from dataclasses import dataclass, field

@dataclass
class SamplingConfig:
    temperature:            float = 1.0
    seed:                   int   = -1
    max_attempts:           int   = 3
    novelty_check:          bool  = True
    silence_check:          bool  = True
    temperature_escalation: float = 1.2   # multiply temp per failed attempt

@dataclass
class TrackPrompt:
    id:             int
    bars:           list[int]
    autoregressive: bool          = False
    ignore:         bool          = False
    attributes:     dict[str,int] = field(default_factory=dict)
    # keys = attribute names (e.g. "note_density"), values = quantized levels

@dataclass
class GenerationRequest:
    tracks: list[TrackPrompt]
    config: SamplingConfig = field(default_factory=SamplingConfig)
```

---

### `midigpt/inference/engine.py`

```python
import midigpt._core as _core
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.attributes.base import AttributeAnalyzer

class InferenceEngine:
    """
    Long-lived. Instantiate once at startup.
    Holds model in memory; sessions are lightweight.
    """

    def __init__(self, model, tokenizer: Tokenizer, analyzer: AttributeAnalyzer):
        self._model     = model
        self._tokenizer = tokenizer
        self._analyzer  = analyzer

    @classmethod
    def from_checkpoint(cls, path: str,
                        analyzer: AttributeAnalyzer | None = None) -> "InferenceEngine":
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]")
        from midigpt.tokenizer.checkpoint import load_checkpoint
        bundle    = load_checkpoint(path)
        model     = torch.jit.load(bundle.model_path, map_location="cpu")
        model.eval()
        tokenizer = Tokenizer(bundle.encoder_config, analyzer)
        return cls(model, tokenizer, analyzer or AttributeAnalyzer.default())

    def session(self, score: "Score",
                request: "GenerationRequest") -> "SamplingSession":
        from midigpt.inference.session import SamplingSession
        return SamplingSession(self, score, request)

    def realtime_session(self, config: "RealtimeConfig") -> "RealtimeSession":
        from midigpt.inference.realtime_session import RealtimeSession
        return RealtimeSession(self, config)
```

---

### `midigpt/inference/session.py`

```python
import copy
import torch
import midigpt._core as _core
from midigpt._types import Score
from midigpt._converters import to_cpp, from_cpp
from midigpt.inference.config import GenerationRequest, SamplingConfig

class SamplingSession:
    """Short-lived. One per generation call. Use as context manager."""

    def __init__(self, engine: "InferenceEngine",
                 score: Score, request: GenerationRequest):
        self._engine  = engine
        self._score   = score
        self._request = request

    def __enter__(self): return self
    def __exit__(self, *_): pass  # Python GC handles cleanup; CM for explicitness

    def run(self) -> Score:
        mask    = self._build_selection_mask()
        planner = _core.StepPlanner(mask, self._engine._tokenizer._vocab.config())
        score   = copy.deepcopy(self._score)
        for step in planner.plan():
            score = self._run_step(score, step)
        return score

    # ── internals ──────────────────────────────────────────────────────────────

    def _run_step(self, score: Score, step) -> Score:
        cfg         = self._request.config
        temperature = cfg.temperature
        best        = None
        if cfg.seed >= 0:
            torch.manual_seed(cfg.seed)
        for _ in range(cfg.max_attempts):
            candidate = self._sample_step(score, step, temperature)
            if self._is_acceptable(score, candidate, cfg):
                return candidate
            if best is None or self._note_count(candidate) > self._note_count(best):
                best = candidate
            if self._note_count(candidate) == 0:
                temperature *= cfg.temperature_escalation
        return best if best is not None else score

    def _sample_step(self, score: Score, step, temperature: float) -> Score:
        ta, ba = self._engine._tokenizer._compute_attrs(score)
        state  = _core.SessionState(
            to_cpp(score), step,
            self._engine._tokenizer._vocab,
            self._build_constraints(step),
            self._engine._tokenizer._encoder,
            self._engine._tokenizer._decoder,
            ta, ba,
        )
        with torch.no_grad():
            while not state.complete():
                ctx    = torch.tensor([state.context_tokens()], dtype=torch.long)
                logits = self._engine._model(ctx)[0, -1]
                mask   = torch.from_numpy(state.logit_mask())  # numpy bool array
                logits[~mask] = float("-inf")
                probs  = (logits / temperature).softmax(-1)
                token  = torch.multinomial(probs, 1).item()
                state.advance(token)
        return from_cpp(state.result())

    def _is_acceptable(self, original: Score, candidate: Score,
                       cfg: SamplingConfig) -> bool:
        if cfg.silence_check and self._note_count(candidate) == 0:
            return False
        if cfg.novelty_check and self._is_identical(original, candidate):
            return False
        return True

    def _note_count(self, score: Score) -> int:
        return sum(len(b.notes) for t in score.tracks for b in t.bars)

    def _is_identical(self, a: Score, b: Score) -> bool:
        for tp in self._request.tracks:
            for bar_idx in tp.bars:
                ta = a.tracks[tp.id].bars[bar_idx] if tp.id < len(a.tracks) else None
                tb = b.tracks[tp.id].bars[bar_idx] if tp.id < len(b.tracks) else None
                if ta is None or tb is None:
                    continue
                if sorted((n.pitch, n.onset_ticks) for n in ta.notes) != \
                   sorted((n.pitch, n.onset_ticks) for n in tb.notes):
                    return False
        return True

    def _build_selection_mask(self) -> _core.SelectionMask:
        n_tracks = len(self._score.tracks)
        n_bars   = max((len(t.bars) for t in self._score.tracks), default=0)
        selected       = [[False] * n_bars for _ in range(n_tracks)]
        autoregressive = [False] * n_tracks
        ignore         = [False] * n_tracks
        for tp in self._request.tracks:
            if tp.id >= n_tracks:
                continue
            for b in tp.bars:
                if b < n_bars:
                    selected[tp.id][b] = True
            autoregressive[tp.id] = tp.autoregressive
            ignore[tp.id]         = tp.ignore
        return _core.SelectionMask(selected, autoregressive, ignore)

    def _build_constraints(self, step) -> _core.ConstraintGraph:
        graph = _core.ConstraintGraph()
        graph.add_grammar_constraint()
        graph.add_polyphony_constraint(max_polyphony=8)  # configurable default
        for tp in self._request.tracks:
            for attr_name, quantized_val in tp.attributes.items():
                # Look up which TokenType this attribute maps to
                attr = self._engine._analyzer._attrs.get(attr_name)
                if attr is not None:
                    token_type = getattr(_core.TokenType, attr.token_type)
                    graph.add_attribute_value_constraint(token_type, quantized_val)
        return graph
```

---

### `midigpt/training/dataset.py`

```python
import pyarrow as pa
import pyarrow.parquet as pq
from midigpt._types import Score
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.augmentation.base import AugmentationPipeline

_SCORE_SCHEMA = pa.schema([
    pa.field("resolution", pa.int32()),
    pa.field("tempo",      pa.int32()),
    pa.field("tracks", pa.list_(pa.struct([
        pa.field("instrument",  pa.int32()),
        pa.field("track_type",  pa.string()),
        pa.field("bars", pa.list_(pa.struct([
            pa.field("ts_numerator",   pa.int32()),
            pa.field("ts_denominator", pa.int32()),
            pa.field("future",         pa.bool_()),
            pa.field("notes", pa.list_(pa.struct([
                pa.field("pitch",          pa.int32()),
                pa.field("velocity",       pa.int32()),
                pa.field("onset_ticks",    pa.int32()),
                pa.field("duration_ticks", pa.int32()),
                pa.field("delta",          pa.int32()),
            ]))),
        ]))),
    ]))),
])

class DatasetBuilder:
    """Preprocessing: MIDI → Parquet. Run once before training."""

    def build(self, midi_paths: list[str], output_path: str,
              splits: dict[str,float] = {"train": 0.9, "valid": 0.05, "test": 0.05}):
        # 1. Parse MIDI → Score for each file
        # 2. Split into train/valid/test
        # 3. Write Arrow-native Parquet (uses _SCORE_SCHEMA above, NOT JSON strings)
        ...


class MidiGPTDataset:
    """On-the-fly augmentation + tokenization. Safe in DataLoader workers."""

    def __init__(self, parquet_path: str, tokenizer: Tokenizer,
                 augmenter: AugmentationPipeline | None = None,
                 max_seq_len: int = 2048):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]")
        self._data        = hf.load_dataset("parquet", data_files=parquet_path, split="train")
        self._tokenizer   = tokenizer
        self._augmenter   = augmenter
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        score = Score.from_dict(self._data[idx])
        if self._augmenter:
            score = self._augmenter(score)
        tokens = self._tokenizer.encode(score)[:self._max_seq_len]
        return {"input_ids": tokens, "labels": tokens}   # standard CLM
```

---

### `midigpt/server/osc_server.py`

External OSC protocol unchanged (Max MSP compatibility). Internals replaced entirely.

```python
from midigpt.inference.engine import InferenceEngine

class MidiGPTServer:
    """Owns one InferenceEngine and one RealtimeSession per active generation."""

    def __init__(self, engine: InferenceEngine, host: str = "0.0.0.0", port: int = 7400):
        self._engine = engine
        self._host   = host
        self._port   = port

    def start(self) -> None: ...   # spins up OSC dispatcher + gen worker thread
    def stop(self)  -> None: ...

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--port",  type=int, default=7400)
    p.add_argument("--host",  default="0.0.0.0")
    args = p.parse_args()
    engine = InferenceEngine.from_checkpoint(args.ckpt)
    MidiGPTServer(engine, args.host, args.port).start()
```

`protocol.py` — parses/formats all OSC messages (same address patterns as current spec).
`state.py` — `RealtimeState` state machine: `UNINITIALIZED → INITIALIZING → RUNNING → STOPPED`.

---

### `midigpt/__init__.py`

```python
from midigpt._types import Score, Track, Bar, Note
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.augmentation.base import AugmentationPipeline, BaseTransform
from midigpt.tokenizer.tokenizer import Tokenizer

def load_engine(checkpoint: str, **kwargs):
    """Shorthand for InferenceEngine.from_checkpoint()."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError("pip install midigpt[inference]")
    from midigpt.inference.engine import InferenceEngine
    return InferenceEngine.from_checkpoint(checkpoint, **kwargs)

__version__ = "0.1.0"
__all__ = [
    "Score", "Track", "Bar", "Note",
    "AttributeAnalyzer", "AugmentationPipeline", "BaseTransform",
    "Tokenizer", "load_engine",
]
```

---

## Build system

### `cmake/dependencies.cmake`

```cmake
include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(symusic
    GIT_REPOSITORY https://github.com/Yikai-Liao/symusic.git
    GIT_TAG        v0.4.5)     # pinned — update deliberately
FetchContent_MakeAvailable(symusic)

# nlohmann/json — single header, vendored at include/nlohmann/json.hpp
# No FetchContent: committed directly, no network required at build time
```

### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.21)
project(midigpt VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/dependencies.cmake)

add_library(midigpt_core STATIC
    src/cpp/io/midi_reader.cpp
    src/cpp/io/midi_writer.cpp
    src/cpp/tokenizer/encoder_config.cpp
    src/cpp/tokenizer/vocabulary.cpp
    src/cpp/tokenizer/encoder.cpp
    src/cpp/tokenizer/decoder.cpp
    src/cpp/masking/constraint_graph.cpp
    src/cpp/masking/grammar_constraint.cpp
    src/cpp/sampling/step_planner.cpp
    src/cpp/sampling/session_state.cpp
)
target_include_directories(midigpt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include)   # nlohmann lives here
target_link_libraries(midigpt_core PUBLIC symusic)

pybind11_add_module(_core MODULE src/cpp/bindings/lib.cpp)
target_link_libraries(_core PRIVATE midigpt_core)
install(TARGETS _core DESTINATION midigpt)

# Copy extension next to __init__.py for in-tree pytest
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_core>"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/midigpt/$<TARGET_FILE_NAME:_core>")
```

### `pyproject.toml`

```toml
[build-system]
requires      = ["scikit-build-core>=0.9", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name            = "midigpt"
version         = "0.1.0"
requires-python = ">=3.10"
dependencies    = []   # base installs with zero Python deps

[project.optional-dependencies]
inference = ["torch>=2.0"]
osc       = ["midigpt[inference]", "python-osc>=1.8"]
train     = ["midigpt[inference]", "transformers>=4.40",
             "datasets>=2.18", "pyarrow>=15.0", "accelerate>=0.29"]
dev       = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.9"]
all       = ["midigpt[osc,train]"]

[project.scripts]
midigpt-server = "midigpt.server.osc_server:main"

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages   = ["src/python/midigpt"]
```

---

## Testing

### C++ — doctest (vendored single header, zero setup)

Each `.cpp` test file is compiled independently alongside `midigpt_core`.

| File | Tests |
|---|---|
| `test_score.cpp` | Struct construction, copy semantics, note pool indexing |
| `test_vocabulary.cpp` | Encode/decode roundtrips, offset arithmetic, domain sizes, range() |
| `test_encoder.cpp` | Known Score → known token sequence for each token type |
| `test_decoder.cpp` | Known token sequence → known Score, edge cases (empty bars, drums) |
| `test_roundtrip.cpp` | `encode(decode(tokens)) == tokens`, `decode(encode(score)) ≈ score` |
| `test_constraint_graph.cpp` | Each constraint: mask correct for valid/invalid token positions |
| `test_step_planner.cpp` | Known SelectionMask → known GenerationStep list |
| `test_session_state.cpp` | advance() sequence → complete(), result() correct |

### Python — pytest

**`tests/python/conftest.py`** — shared fixtures:
```python
@pytest.fixture
def simple_score() -> Score:
    """One melodic track, four bars, a handful of notes."""
    ...

@pytest.fixture
def two_track_score() -> Score:
    """One melodic + one drum track."""
    ...

@pytest.fixture
def stub_model():
    """Returns uniform random logits — valid for testing session logic."""
    import torch
    class StubModel(torch.nn.Module):
        def forward(self, x):
            vocab = 512   # arbitrary
            return torch.rand(x.shape[0], x.shape[1], vocab)
    return StubModel()

@pytest.fixture
def tokenizer(tmp_path) -> Tokenizer:
    config = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
    return Tokenizer(config)
```

| File | Tests |
|---|---|
| `test_types.py` | Score/Track/Bar/Note construction, `from_midi`/`to_midi` roundtrip |
| `test_converters.py` | `to_cpp`/`from_cpp` roundtrip, note pool integrity |
| `test_attributes.py` | Each attribute: known Score → expected value; `evaluate()` match scoring |
| `test_augmentation.py` | Each transform: output valid Score, drums unaffected by Transpose |
| `test_tokenizer.py` | `encode`→`decode` roundtrip; attribute tokens present in output |
| `test_inference.py` | `SamplingSession.run()` with stub model; novelty/silence checks fire correctly |
| `test_training.py` | Parquet roundtrip; `MidiGPTDataset.__getitem__` returns correct shape |
| `test_server.py` | OSC message parsing; protocol state machine transitions |

---

## Implementation sequence

| Step | Deliverable | Tests unlocked |
|---|---|---|
| 1 | C++ `types.h` + `score.h` | `test_score.cpp` |
| 2 | C++ `MidiReader` + `MidiWriter` (symusic integration) | `test_io.cpp` |
| 3 | C++ `EncoderConfig` + `Vocabulary` | `test_vocabulary.cpp` |
| 4 | C++ `Encoder` + `Decoder` | `test_encoder.cpp`, `test_decoder.cpp`, `test_roundtrip.cpp` |
| 5 | C++ `ConstraintGraph` + all constraints | `test_constraint_graph.cpp` |
| 6 | C++ `StepPlanner` + `SessionState` | `test_step_planner.cpp`, `test_session_state.cpp` |
| 7 | pybind11 bindings + Python `_types.py` + `_converters.py` | `test_types.py`, `test_converters.py` |
| 8 | Python `AttributeAnalyzer` + all 8 attributes | `test_attributes.py` |
| 9 | Python `AugmentationPipeline` + all 5 transforms | `test_augmentation.py` |
| 10 | Python `Tokenizer` (orchestrator) | `test_tokenizer.py` |
| 11 | Python `InferenceEngine` + `SamplingSession` | `test_inference.py` (stub model) |
| 12 | Python `DatasetBuilder` + `MidiGPTDataset` | `test_training.py` |
| 13 | Python `MidiGPTServer` (OSC rewrite) | `test_server.py` |
| 14 | `cibuildwheel` CI matrix | Wheel builds on all platforms |

Steps 8 and 9 are independent after step 7 and can proceed in parallel. Steps 12 and 13 are independent after step 11 and can proceed in parallel. Every step before the next is fully tested and merged before starting the next.
