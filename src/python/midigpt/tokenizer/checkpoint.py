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
