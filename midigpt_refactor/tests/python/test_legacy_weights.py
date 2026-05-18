import json
from pathlib import Path

from midigpt_refactor import _core
from midigpt_refactor.tokenizer import Tokenizer


def _vocab_size_for(config_name: str) -> int:
    config_path = (Path(__file__).parent.parent.parent.parent
                   / "models" / config_name)
    assert config_path.exists(), f"{config_name} not found"
    with open(config_path, "r") as f:
        data = f.read()
    config = _core.EncoderConfig.from_json(data)
    # Go through Tokenizer so attribute-control token domains are appended
    # (sizes live in the Python attribute classes).
    tok = Tokenizer(config)
    return tok.vocab_size()


def test_ghost_encoder_vocab_size():
    assert _vocab_size_for("ghost_config.json") == 650


def test_expressive_encoder_vocab_size():
    assert _vocab_size_for("expressive_config.json") == 840


def test_yellow_encoder_vocab_size():
    assert _vocab_size_for("yellow_config.json") == 647
