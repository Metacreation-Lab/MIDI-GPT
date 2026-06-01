from midigpt._types import Score, Track, Bar, Note

import os
from ._core import set_verbosity, LogLevel

# Initialize logging level from environment variable
_env_log_level = os.environ.get("MIDIGPT_LOG_LEVEL")
if _env_log_level is not None:
    try:
        if _env_log_level.isdigit():
            set_verbosity(int(_env_log_level))
        else:
            set_verbosity(getattr(LogLevel, _env_log_level.upper()))
    except (ValueError, AttributeError):
        pass

__version__ = "0.2.1"
__all__ = [
    "Score", "Track", "Bar", "Note",
]
