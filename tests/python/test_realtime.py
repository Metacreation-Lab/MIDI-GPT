"""
Tests for tension-aware models (specter, oracle) and the realtime OSC server.

Marks:
  slow      — requires real model bundles in models/
  inference — requires torch
"""
import json
import pathlib
import pytest

pytest.importorskip("torch", reason="inference extra not installed")
pythonosc = pytest.importorskip("pythonosc", reason="realtime extra not installed")

import torch

import midigpt._core as _core
from midigpt._types import Score, Track, Bar, Note
from midigpt.inference import InferenceEngine, SamplingConfig, TrackPrompt, GenerationRequest
from midigpt.inference.model import GPT2Config, GPT2LMHeadModel


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SPECTER_BUNDLE = pathlib.Path("models/specter_bundle.pt")
ORACLE_BUNDLE  = pathlib.Path("models/oracle_bundle.pt")

ORACLE_ENC_CFG = {
    "resolution": 12,
    "decode_resolution": 1920,
    "use_microtiming": True,
    "supports_infill": True,
    "token_domains": [
        {"type": "PieceStart",        "domain_size": 2},
        {"type": "NumBars",           "domain_size": 4},
        {"type": "Bar",               "domain_size": 1},
        {"type": "BarEnd",            "domain_size": 1},
        {"type": "TimeSig",           "domain_size": 36},
        {"type": "Track",             "domain_size": 2},
        {"type": "TrackEnd",          "domain_size": 1},
        {"type": "Instrument",        "domain_size": 109},
        {"type": "NoteOnset",         "domain_size": 128},
        {"type": "NoteDuration",      "domain_size": 96},
        {"type": "TimeAbsolutePos",   "domain_size": 192},
        {"type": "FillInPlaceholder", "domain_size": 1},
        {"type": "FillInStart",       "domain_size": 1},
        {"type": "FillInEnd",         "domain_size": 1},
        {"type": "MaskBar",           "domain_size": 1},
        {"type": "MinNoteDuration",   "domain_size": 6},
        {"type": "MaxNoteDuration",   "domain_size": 6},
        {"type": "MinPolyphony",      "domain_size": 10},
        {"type": "MaxPolyphony",      "domain_size": 10},
        {"type": "NoteDensity",       "domain_size": 10},
        {"type": "Tension",           "domain_size": 10},
        {"type": "TensionDrum",       "domain_size": 10},
        {"type": "VelocityLevel",     "domain_size": 32},
    ],
    "time_signatures": [
        "4/4","3/4","2/4","6/8","2/2","1/4","6/4","3/8","5/4","4/2","1/8",
        "3/2","9/8","5/8","7/8","12/8","8/4","7/4","4/8","3/1","1/2","8/8",
        "11/8","2/8","6/2","9/4","2/1","9/16","12/4","10/4","13/16","15/16",
        "17/16","1/16","10/8","16/4"
    ],
    "num_bars_map": [4, 8, 12, 16],
    "track_map": [10, 11],
}

ORACLE_VOCAB = sum(d["domain_size"] for d in ORACLE_ENC_CFG["token_domains"])  # 670


def _make_score(n_tracks: int = 2, n_bars: int = 4) -> Score:
    notes = [Note(pitch=60 + i, velocity=80, onset_ticks=i * 120, duration_ticks=120)
             for i in range(4)]
    bars  = [Bar(notes=notes[:], ts_numerator=4, ts_denominator=4) for _ in range(n_bars)]
    return Score(tracks=[
        Track(bars=[Bar(notes=notes[:], ts_numerator=4, ts_denominator=4) for _ in range(n_bars)],
              instrument=i * 30, track_type="melodic")
        for i in range(n_tracks)
    ])


def _tiny_oracle_engine() -> InferenceEngine:
    """Create an InferenceEngine with oracle's encoder config but a tiny GPT-2."""
    enc_cfg = _core.EncoderConfig.from_json(json.dumps(ORACLE_ENC_CFG))
    gpt_cfg = GPT2Config(vocab_size=ORACLE_VOCAB, n_positions=256,
                         n_embd=16, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(gpt_cfg)
    model.encoder_config = ORACLE_ENC_CFG
    model.eval()
    from midigpt.tokenizer import Tokenizer
    from midigpt.attributes.base import AttributeAnalyzer
    tokenizer = Tokenizer(enc_cfg)
    analyzer  = AttributeAnalyzer.from_config(enc_cfg)
    return InferenceEngine(model, tokenizer, analyzer)


# ---------------------------------------------------------------------------
# Specter: tension token domains — inference smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.inference
def test_specter_tension_inference():
    """Specter bundle (Tension + TensionDrum token domains) loads and generates."""
    if not SPECTER_BUNDLE.exists():
        pytest.skip("models/specter_bundle.pt not found")

    engine = InferenceEngine.from_checkpoint(str(SPECTER_BUNDLE))
    enc_cfg_dict = torch.load(str(SPECTER_BUNDLE), map_location="cpu",
                              weights_only=False)["encoder_config"]

    assert any(d["type"] == "Tension"     for d in enc_cfg_dict["token_domains"])
    assert any(d["type"] == "TensionDrum" for d in enc_cfg_dict["token_domains"])

    score = _make_score(n_tracks=1, n_bars=4)
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[2, 3])],
        config=SamplingConfig(
            max_attempts=1,
            silence_check=False,
            novelty_check=False,
        ),
    )
    with engine.session(score, req) as session:
        out = session.run()

    assert isinstance(out, Score)
    assert len(out.tracks) == 1


@pytest.mark.slow
@pytest.mark.inference
def test_specter_tension_token_domains_vocab():
    """Specter vocab size matches sum of token_domains (667)."""
    if not SPECTER_BUNDLE.exists():
        pytest.skip("models/specter_bundle.pt not found")

    ckpt = torch.load(str(SPECTER_BUNDLE), map_location="cpu", weights_only=False)
    vocab_from_weights = ckpt["state_dict"]["transformer.wte.weight"].shape[0]
    vocab_from_domains = sum(d["domain_size"] for d in ckpt["encoder_config"]["token_domains"])

    assert vocab_from_weights == 667
    assert vocab_from_domains == vocab_from_weights


# ---------------------------------------------------------------------------
# Oracle: realtime server smoke tests (tiny model, no heavy bundle required)
# ---------------------------------------------------------------------------

def test_oracle_server_session_lifecycle():
    """MidiGPTServer handles session_init → track_create → session_start correctly."""
    from midigpt.osc.server import MidiGPTServer

    engine = _tiny_oracle_engine()

    server = MidiGPTServer.__new__(MidiGPTServer)
    server._ckpt        = "tiny-oracle"
    server._listen_port = 0
    server._max_attempts = 3
    server._state       = "UNINITIALIZED"
    server._state_lock  = __import__("threading").Lock()
    server._piece       = None
    server._client      = None
    server._client_lock = __import__("threading").Lock()
    from midigpt.osc.gen import PARAM_DEFAULTS
    server._params      = dict(PARAM_DEFAULTS)
    server._params["max_attempts"] = 3
    server._once_params = {}
    server._params_lock = __import__("threading").Lock()
    server._engine      = engine
    import queue, threading
    server._gen_queue   = queue.Queue(maxsize=1)
    server._gen_thread  = threading.Thread(
        target=server._gen_worker, daemon=True, name="midigpt-gen-test"
    )
    server._gen_thread.start()

    addr = ("127.0.0.1", 7401)

    server.handle_session_init(addr, "/midigpt/session/init", "test-session")
    assert server._state == "INITIALIZING"
    assert server._piece is not None

    server.handle_track_create(addr, "/midigpt/track/create", 0, 0, 10, 0)
    server.handle_track_create(addr, "/midigpt/track/create", 1, 40, 10, 1)

    assert server._piece.has_agent()
    assert server._piece.has_conditioning_tracks()
    assert server._piece.num_tracks == 2

    server._params["buffer_bars"] = 4

    server.handle_session_start(addr, "/midigpt/session/start")
    assert server._state == "RUNNING"


def test_oracle_server_note_and_bar():
    """handle_note + handle_bar_end correctly populate PieceState."""
    from midigpt.osc.server import MidiGPTServer

    engine = _tiny_oracle_engine()

    server = MidiGPTServer.__new__(MidiGPTServer)
    server._ckpt        = "tiny-oracle"
    server._listen_port = 0
    server._max_attempts = 3
    server._state       = "RUNNING"
    server._state_lock  = __import__("threading").Lock()
    server._client      = None
    server._client_lock = __import__("threading").Lock()
    from midigpt.osc.gen import PARAM_DEFAULTS
    server._params      = dict(PARAM_DEFAULTS)
    server._params["max_attempts"] = 3
    server._once_params = {}
    server._params_lock = __import__("threading").Lock()
    server._engine      = engine
    import queue, threading
    server._gen_queue   = queue.Queue(maxsize=1)
    server._gen_thread  = threading.Thread(
        target=server._gen_worker, daemon=True, name="midigpt-gen-test2"
    )
    server._gen_thread.start()

    from midigpt.osc.piece_state import PieceState
    resolution = 12
    server._piece = PieceState(resolution=resolution)
    server._piece.create_track(0, 0,  10, is_agent=False)
    server._piece.create_track(1, 40, 10, is_agent=True)

    addr = ("127.0.0.1", 7401)

    server.handle_note(addr, "/midigpt/note", 0, 60, 80, 0.0, 0.5, 0)
    server.handle_note(addr, "/midigpt/note", 0, 64, 75, 0.25, 0.25, 0)

    server.handle_bar_end(addr, "/midigpt/bar/end", 0, 4, 4)

    assert server._piece.bars_completed == 1
    cond_track = [t for t in server._piece._tracks.values() if not t.is_agent][0]
    assert len(cond_track.bars[0]["events"]) == 2


@pytest.mark.slow
@pytest.mark.inference
def test_oracle_bundle_loads():
    """Oracle bundle (Tension + MaskBar) loads and encoder config is consistent."""
    if not ORACLE_BUNDLE.exists():
        pytest.skip("models/oracle_bundle.pt not found")

    ckpt = torch.load(str(ORACLE_BUNDLE), map_location="cpu", weights_only=False)
    enc  = ckpt["encoder_config"]
    sd   = ckpt["state_dict"]

    vocab_weights = sd["transformer.wte.weight"].shape[0]
    vocab_domains = sum(d["domain_size"] for d in enc["token_domains"])

    assert vocab_weights == 670
    assert vocab_domains == vocab_weights
    assert any(d["type"] == "Tension"     for d in enc["token_domains"])
    assert any(d["type"] == "TensionDrum" for d in enc["token_domains"])
    assert any(d["type"] == "MaskBar"     for d in enc["token_domains"])

    engine = InferenceEngine.from_checkpoint(str(ORACLE_BUNDLE))
    score  = _make_score(n_tracks=2, n_bars=4)
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[2, 3]),
            TrackPrompt(id=1, bars=[], ignore=True),
        ],
        config=SamplingConfig(max_attempts=1, silence_check=False, novelty_check=False),
    )
    with engine.session(score, req) as session:
        out = session.run()

    assert isinstance(out, Score)
