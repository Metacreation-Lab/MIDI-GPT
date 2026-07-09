import os

from midigpt._types import Bar, Note, Score, Track

from ._core import LogLevel, set_verbosity

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

__version__ = "0.3.2"
__all__ = [
    "Bar",
    "Note",
    "Score",
    "Track",
]
