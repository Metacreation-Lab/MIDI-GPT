"""midigpt — MIDI music generation via GPT-2 Transformer.

C++/Python hybrid package.  The compiled C++ extension ``_midigpt`` is
re-exported here so callers use ``import midigpt`` as normal.

Torch loading order
-------------------
When the extension is built with LibTorch (MIDIGPT_NO_TORCH is OFF), the
torch shared libraries must already be mapped into the process before
``_midigpt.so`` is dlopen-ed.  Importing ``torch`` here guarantees the
correct loading order regardless of what the caller does.
"""
from __future__ import annotations

# Side-effect import: loads torch's shared libraries into the process so
# _midigpt.so can resolve libtorch symbols at dlopen time.
# In NO_TORCH builds torch is not installed; the ImportError is silenced.
try:
    import torch as _torch  # noqa: F401
    has_torch: bool = True
except ImportError:
    has_torch: bool = False

from ._midigpt import *          # noqa: F401, F403 — re-export full public API
from ._midigpt import version    # ensure `midigpt.version` is explicitly available

__version__: str = version()
