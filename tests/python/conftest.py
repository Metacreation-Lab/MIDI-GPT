"""Shared fixtures for the Python test suite.

Goals:
- No production-code "tweaks" to make tests pass.
- Synthetic, deterministic in-memory fixtures wherever possible.
- Real shipped configs (models/*_config.json) for any test that depends on
  the canonical encoder spec — never duplicate that spec in test code.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch

import midigpt._core as _core
from midigpt._types import Bar, Note, Score, Track
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.tokenizer.tokenizer import Tokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MIDI_DIR = REPO_ROOT / "tests" / "midi"


# --------------------------------------------------------------------------- #
#  Encoder configs (real shipped JSON — single source of truth for spec)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ghost_config_json() -> str:
    return (MODELS_DIR / "ghost_config.json").read_text()


@pytest.fixture
def ghost_config(ghost_config_json) -> _core.EncoderConfig:
    """Fresh EncoderConfig per test (tests may mutate it)."""
    return _core.EncoderConfig.from_json(ghost_config_json)


@pytest.fixture
def ghost_analyzer(ghost_config) -> AttributeAnalyzer:
    return AttributeAnalyzer.from_config(ghost_config)


@pytest.fixture
def ghost_tokenizer(ghost_config, ghost_analyzer) -> Tokenizer:
    return Tokenizer(ghost_config, ghost_analyzer)


# --------------------------------------------------------------------------- #
#  Score builders
# --------------------------------------------------------------------------- #
def make_bar(notes=None, ts_num=4, ts_den=4, beat_length=4.0) -> Bar:
    return Bar(
        notes=list(notes or []), ts_numerator=ts_num, ts_denominator=ts_den, beat_length=beat_length
    )


def make_note(pitch=60, vel=80, onset=0, dur=120, delta=0) -> Note:
    return Note(pitch=pitch, velocity=vel, onset_ticks=onset, duration_ticks=dur, delta=delta)


def melodic_track(n_bars=4, notes_per_bar=4, base_pitch=60, res=12, instrument=0) -> Track:
    """A simple melodic track: ascending C-major-ish line, one note per beat."""
    bars = []
    for b in range(n_bars):
        notes = []
        for i in range(notes_per_bar):
            onset = i * (res * 4 // notes_per_bar)
            notes.append(
                make_note(
                    pitch=base_pitch + (b + i) % 12,
                    vel=80,
                    onset=onset,
                    dur=res,
                )
            )
        bars.append(make_bar(notes))
    return Track(bars=bars, instrument=instrument, track_type="melodic")


def drum_track(n_bars=4, res=12) -> Track:
    """Simple kick-on-1, snare-on-3 drum track."""
    bars = []
    for _ in range(n_bars):
        bars.append(
            make_bar(
                [
                    make_note(pitch=36, onset=0, dur=res // 2),
                    make_note(pitch=38, onset=2 * res, dur=res // 2),
                ]
            )
        )
    return Track(bars=bars, instrument=0, track_type="drum")


@pytest.fixture
def simple_score() -> Score:
    """1 melodic track, 4 bars, 4 notes/bar."""
    return Score(tracks=[melodic_track(n_bars=4)], resolution=12, tempo=500000)


@pytest.fixture
def two_track_score() -> Score:
    """Melodic + drum, 4 bars each."""
    return Score(
        tracks=[melodic_track(n_bars=4), drum_track(n_bars=4)],
        resolution=12,
        tempo=500000,
    )


@pytest.fixture
def empty_bars_score() -> Score:
    """1 melodic track, 4 EMPTY bars (silent throughout)."""
    return Score(
        tracks=[Track(bars=[make_bar() for _ in range(4)], instrument=0, track_type="melodic")],
        resolution=12,
        tempo=500000,
    )


# --------------------------------------------------------------------------- #
#  Tiny synthetic GPT-2 (real forward path, random init, deterministic seed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_gpt2_config(ghost_tokenizer) -> GPT2Config:
    """Vocab size derived from the real tokenizer so model+vocab agree."""
    return GPT2Config(
        vocab_size=ghost_tokenizer.vocab_size(),
        n_positions=512,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )


@pytest.fixture
def tiny_gpt2(tiny_gpt2_config) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(tiny_gpt2_config)
    # Conv1D uses torch.empty() — initialize so forward doesn't produce NaN.
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Fake ModelBase — canned logits, full ModelBase surface, no torch.nn needed
# --------------------------------------------------------------------------- #
class FakeModel:
    """Returns deterministic logits, implements ModelBase.

    `logit_fn(input_ids, past_kv) -> Tensor[B, T, V]` lets tests inject
    arbitrary logit shapes per call. Default: uniform zeros (uniform draws).
    """

    arch = "fake"
    encoder_config = None

    def __init__(
        self,
        vocab_size: int,
        n_positions: int = 512,
        n_layer: int = 2,
        n_head: int = 2,
        head_dim: int = 8,
        logit_fn=None,
    ):
        self.vocab_size = vocab_size
        self._n_positions = n_positions
        self._n_layer = n_layer
        self._n_head = n_head
        self._head_dim = head_dim
        self.calls: list[dict] = []
        self._logit_fn = logit_fn or (
            lambda ids, kv: torch.zeros(ids.shape[0], ids.shape[1], vocab_size)
        )

    def __call__(self, input_ids, past_kv=None, **kwargs):
        return self.forward(input_ids, past_kv, **kwargs)

    def forward(self, input_ids, past_kv=None, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "past_len": self.kv_length(past_kv),
                "kwargs": {k: v for k, v in kwargs.items()},
            }
        )
        logits = self._logit_fn(input_ids, past_kv)
        # Append T zero positions to each layer's KV
        T = input_ids.shape[1]
        present = []
        for k, v in past_kv or self.make_empty_kv():
            new_k = torch.cat([k, torch.zeros(1, self._n_head, T, self._head_dim)], dim=2)
            new_v = torch.cat([v, torch.zeros(1, self._n_head, T, self._head_dim)], dim=2)
            present.append((new_k, new_v))
        return logits, tuple(present)

    def make_empty_kv(self):
        return tuple(
            (
                torch.zeros(1, self._n_head, 0, self._head_dim),
                torch.zeros(1, self._n_head, 0, self._head_dim),
            )
            for _ in range(self._n_layer)
        )

    def kv_length(self, past_kv) -> int:
        if past_kv is None or len(past_kv) == 0:
            return 0
        return int(past_kv[0][0].shape[2])

    def kv_null_positions(self, past_kv, spans):
        if past_kv is None or not spans:
            return
        for k, v in past_kv:
            for s, e in spans:
                k[:, :, s:e, :] = -1e4
                v[:, :, s:e, :] = 0.0

    def max_context(self) -> int:
        return self._n_positions

    def parameters(self):
        return iter([])


@pytest.fixture
def fake_model_factory(ghost_tokenizer):
    """Returns a builder so tests can customize logit_fn."""
    vocab = ghost_tokenizer.vocab_size()

    def _make(logit_fn=None, **kw):
        return FakeModel(vocab_size=vocab, logit_fn=logit_fn, **kw)

    return _make


# --------------------------------------------------------------------------- #
#  Packed-bundle on disk (synthetic) — for checkpoint / engine tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def packed_bundle_path(tmp_path, tiny_gpt2, ghost_config_json) -> pathlib.Path:
    bundle = tmp_path / "tiny.safetensors"
    tiny_gpt2.save_pretrained(str(bundle), encoder_config=json.loads(ghost_config_json))
    return bundle


# --------------------------------------------------------------------------- #
#  MIDI fixture
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_midi_path() -> pathlib.Path:
    p = MIDI_DIR / "Aicha.mid"
    if not p.exists():
        pytest.skip(f"MIDI fixture missing: {p}")
    return p


# --------------------------------------------------------------------------- #
#  Real-MIDI fixtures: exercise components on realistic, full-length data
# --------------------------------------------------------------------------- #
#
# `tests/midi/` ships a curated mix:
#   - pop songs (multi-track, many bars):  Aicha, Funkytown, Mr. Blue Sky, ...
#   - classical/piano (long, dense):       Maestro_*, POP909_*, Etude No. 4
#   - edge-case files (named edge_*.mid):  cross-bar sustain, dense polyphony,
#                                          drums + melodic, time-sig change,
#                                          very long note
#   - empty.mid                             zero-content sentinel
#
# Tests SHOULD prefer these over hand-rolled in-memory scores whenever the
# component under test cares about realistic structure (many bars, multiple
# tracks, drums + melodic, sustained notes, time-sig changes, etc.).

REAL_POP_FILES = [
    "Aicha.mid",
    "Funkytown.mid",
    "Mr. Blue Sky.mid",
    "All The Small Things.mid",
    "I Gotta Feeling.mid",
]
REAL_PIANO_FILES = [
    "Maestro_1.mid",
    "Maestro_2.mid",
    "POP909_008.mid",
    "POP909_010.mid",
    "6338816_Etude No. 4.mid",
]
EDGE_FILES = [
    "edge_cross_bar_sustain.mid",
    "edge_dense_polyphony.mid",
    "edge_drums_and_melodic.mid",
    "edge_time_sig_change.mid",
    "edge_very_long_note.mid",
]


def _midi(name: str) -> pathlib.Path:
    p = MIDI_DIR / name
    if not p.exists():
        pytest.skip(f"MIDI fixture missing: {p}")
    return p


@pytest.fixture
def midi_dir() -> pathlib.Path:
    return MIDI_DIR


@pytest.fixture(params=REAL_POP_FILES)
def real_pop_midi_path(request) -> pathlib.Path:
    return _midi(request.param)


@pytest.fixture(params=REAL_PIANO_FILES)
def real_piano_midi_path(request) -> pathlib.Path:
    return _midi(request.param)


@pytest.fixture(params=REAL_POP_FILES + REAL_PIANO_FILES)
def real_midi_path(request) -> pathlib.Path:
    """Parametrized across a mix of pop and piano files — use for invariants
    that should hold on any realistic MIDI."""
    return _midi(request.param)


@pytest.fixture(params=EDGE_FILES)
def edge_midi_path(request) -> pathlib.Path:
    return _midi(request.param)


@pytest.fixture
def empty_midi_path() -> pathlib.Path:
    return _midi("empty.mid")


def _trim_to_window(score: Score, n_bars: int = 16) -> Score:
    """Trim every track to the first `n_bars` bars so the score fits the
    encoder's num_bars_map = [4, 8, 12, 16] window constraint."""
    for t in score.tracks:
        t.bars = t.bars[:n_bars]
    return score


@pytest.fixture
def real_score(sample_midi_path) -> Score:
    """A realistic Score (Aicha) trimmed to 16 bars — multi-track, dense.
    Use when you want realistic data without parametrization overhead."""
    return _trim_to_window(Score.from_midi(str(sample_midi_path)), 16)


@pytest.fixture
def real_score_untrimmed(sample_midi_path) -> Score:
    """Full Aicha score with all bars/tracks — use only with components
    that can handle arbitrary lengths (Score IO, MIDI roundtrip, etc.)."""
    return Score.from_midi(str(sample_midi_path))


@pytest.fixture
def long_piano_score() -> Score:
    """A dense single-instrument score (Maestro_1, 16 bars) — good for
    stress-testing encoders, attribute analyzers, and step planners."""
    return _trim_to_window(Score.from_midi(str(_midi("Maestro_1.mid"))), 16)


@pytest.fixture
def edge_drums_and_melodic_score() -> Score:
    return Score.from_midi(str(_midi("edge_drums_and_melodic.mid")))


# --------------------------------------------------------------------------- #
#  Parquet data — built on-the-fly from the shipped MIDI fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def training_parquet(tmp_path_factory) -> pathlib.Path:
    """Session-scoped parquet built from the MIDI files in tests/midi/.

    Each row holds the raw MIDI bytes plus the three metadata columns that
    MidiGPTDataset uses for its two-phase filter.  No external data needed:
    the parquet is reconstructed every test session from committed fixtures.

    Falls back to MIDIGPT_TEST_PARQUET if set, for benchmarking with a full
    GigaMIDI shard on a development machine.
    """
    env = os.environ.get("MIDIGPT_TEST_PARQUET", "")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not available — cannot build training parquet fixture")

    from midigpt._types import Score

    midi_files = sorted(
        f for f in MIDI_DIR.glob("*.mid") if f.stem != "empty"
    )
    if not midi_files:
        pytest.skip("No MIDI files found in tests/midi/")

    music_col: list[bytes] = []
    num_tracks_col: list[int] = []
    total_notes_col: list[int] = []
    beats_col: list[float] = []

    for midi_path in midi_files:
        try:
            raw = midi_path.read_bytes()
            score = Score.from_bytes(raw)
            if not score.tracks:
                continue
            n_notes = sum(
                len(bar.notes) for t in score.tracks for bar in t.bars
            )
            if n_notes == 0:
                continue
            max_bars = max(len(t.bars) for t in score.tracks)
            music_col.append(raw)
            num_tracks_col.append(len(score.tracks))
            total_notes_col.append(n_notes)
            # Conservative: 4 beats per bar
            beats_col.append(float(max_bars * 4))
        except Exception:
            continue

    if not music_col:
        pytest.skip("No usable MIDI files could be parsed from tests/midi/")

    table = pa.table(
        {
            "music": pa.array(music_col, type=pa.large_binary()),
            "num_tracks": pa.array(num_tracks_col, type=pa.int32()),
            "total_notes": pa.array(total_notes_col, type=pa.int64()),
            "loop_duration_beats": pa.array(beats_col, type=pa.float64()),
        }
    )
    out = tmp_path_factory.mktemp("parquet") / "midi_fixtures.parquet"
    pq.write_table(table, str(out))
    return out
