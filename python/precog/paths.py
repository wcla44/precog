"""XDG locations (used on macOS too, like most CLI tools)."""
import os
from pathlib import Path


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default) / "precog"


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config")  # settings
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share")  # trained gaze model
CACHE_DIR = _xdg("XDG_CACHE_HOME", ".cache")  # debug screenshots
