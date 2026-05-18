"""Shared utilities for comparison tests between original midigpt and midigpt_refactor.

Both packages are installed side-by-side:
  - `midigpt` (original C++/pybind11 module from /src/python/midigpt)
  - `midigpt_refactor` (refactored module from /midigpt_refactor/src/python/midigpt_refactor)

Tests compare token sequences, decode roundtrips, and runtime speed.
"""
from __future__ import annotations
import contextlib, os, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIDI_DIR = Path(__file__).parent / "midi"
YELLOW_CFG = REPO_ROOT / "models" / "yellow_config.json"


@contextlib.contextmanager
def silence_stdio():
    """The original C++ encoder writes parsed-piece JSON to stdout.
    Redirect FD-level so it doesn't pollute test output."""
    # Use FD 1/2 directly — under pytest capture, sys.stdout isn't a real file.
    fd_out = 1
    fd_err = 2
    saved_out = os.dup(fd_out)
    saved_err = os.dup(fd_err)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd_out)
    os.dup2(devnull, fd_err)
    try:
        yield
    finally:
        os.dup2(saved_out, fd_out); os.dup2(saved_err, fd_err)
        os.close(saved_out); os.close(saved_err); os.close(devnull)


def midi_files(filter_predicate=None):
    """Return sorted list of MIDI files in the comparison/midi/ folder."""
    files = sorted(MIDI_DIR.glob("*.mid"))
    if filter_predicate:
        files = [f for f in files if filter_predicate(f)]
    return files


@pytest.fixture(scope="session")
def yellow_config_text():
    return YELLOW_CFG.read_text()


@pytest.fixture(scope="session")
def yellow_orig_encoder():
    import midigpt
    with silence_stdio():
        enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
    return enc


@pytest.fixture(scope="session")
def yellow_ref_components(yellow_config_text):
    """Refactored Vocabulary + Encoder + Decoder using yellow_config.
    Routes through Tokenizer so attribute-control token domains are appended."""
    import midigpt_refactor._core as _core
    from midigpt_refactor.tokenizer import Tokenizer
    cfg = _core.EncoderConfig.from_json(yellow_config_text)
    tok = Tokenizer(cfg)
    vocab = tok._vocab
    enc = _core.Encoder(vocab)
    dec = _core.Decoder(vocab)
    return cfg, vocab, enc, dec


def midi_param(p: Path):
    return pytest.param(str(p), id=p.stem)


def pretty_ref(vocab, token: int) -> str:
    """Pretty-print a refactored token: 'TokenType:value'."""
    try:
        tt, val = vocab.decode(token)
        return f"{str(tt).split('.')[-1]}:{val}"
    except Exception:
        return f"?:{token}"


def pretty_orig(orig_enc, token: int) -> str:
    """Pretty-print a token via the original encoder (with stdout silenced)."""
    with silence_stdio():
        try:
            return orig_enc.pretty(token)
        except Exception:
            return f"?:{token}"


def diff_report(label_a, tokens_a, pretty_a_fn,
                label_b, tokens_b, pretty_b_fn,
                context: int = 5, max_show: int = 30) -> str:
    """Return a human-readable diff of two token sequences with pretty names."""
    lines = [
        f"  {label_a} len={len(tokens_a)}",
        f"  {label_b} len={len(tokens_b)}",
    ]
    n = min(len(tokens_a), len(tokens_b))
    first_diff = next((i for i in range(n) if tokens_a[i] != tokens_b[i]), None)
    if first_diff is None and len(tokens_a) == len(tokens_b):
        return "  (sequences are equal)"
    if first_diff is None:
        first_diff = n
    lo = max(0, first_diff - context)
    hi = min(max(len(tokens_a), len(tokens_b)), first_diff + context + 1)
    lines.append(f"  first diff @ index {first_diff}, showing [{lo}:{hi}]")
    for i in range(lo, hi):
        a = tokens_a[i] if i < len(tokens_a) else None
        b = tokens_b[i] if i < len(tokens_b) else None
        ap = pretty_a_fn(a) if a is not None else "—"
        bp = pretty_b_fn(b) if b is not None else "—"
        marker = " " if a == b else "*"
        lines.append(f"   {marker} [{i:4d}] {label_a}={a!s:>4} ({ap})  |  {label_b}={b!s:>4} ({bp})")
    if max(len(tokens_a), len(tokens_b)) > hi:
        lines.append(f"   ... +{max(len(tokens_a), len(tokens_b)) - hi} more")
    return "\n".join(lines)


# Fixture to set C++ logging level to ERROR for the test session
@pytest.fixture(autouse=True, scope="session")
def set_cpp_logging_level():
    import midigpt_refactor._core as _core
    _core.set_verbosity(_core.LogLevel.ERROR)
