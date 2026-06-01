"""
Shared utilities for MIDI-GPT: cache directory resolution and cached index files.

Cache root is controlled by the MIDIGPT_CACHE env var (default: ~/.midigpt).
Usable by both training (dataset filter indices) and inference (model files).
"""

import hashlib
import os
from pathlib import Path


def cache_dir() -> Path:
    """Return (and create) the MIDI-GPT cache root directory."""
    root = Path(os.environ.get("MIDIGPT_CACHE", "~/.midigpt")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def cached_indices(key: str) -> Path:
    """Path to a .npy file for storing a list of valid dataset indices."""
    return cache_dir() / f"valid_{key}.npy"


def file_cache_key(path: str, **extra) -> str:
    """Stable MD5 key derived from a file's path, size, mtime, and any extra kwargs."""
    stat = Path(path).stat()
    raw = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    for k, v in sorted(extra.items()):
        raw += f":{k}={v}"
    return hashlib.md5(raw.encode()).hexdigest()
