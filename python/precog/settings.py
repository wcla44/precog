"""Persistent user settings (~/.config/precog/config.json, chmod 600: it may hold an API key)."""
from __future__ import annotations

import json
import os

from .paths import CONFIG_DIR

PATH = CONFIG_DIR / "config.json"

DEFAULTS = {
    "agent": True,  # ask the LLM for suggestions
    "dot": True,  # show the gaze ring
    "gesture": True,  # require a pinch to accept
    "learn": True,  # learn from mouse clicks
    "dwell": 1.8,  # fixation seconds before asking
    "interval": 12.0,  # min seconds between LLM calls
    # gestures (no LLM): open hand = voice typing into the window you look at, fist = send, swipe left = cancel
    "voice": True,
    "stt_model": "base",  # faster-whisper size: tiny | base | small | large-v3-turbo
    "stt_language": "",  # "" = auto-detect, else e.g. "en", "fr"
    "click_focus": False,  # also click where you look (puts the caret in that field; gaze is ~2-3cm accurate)
    "mirror": False,  # flip swipe direction (if your camera image is mirrored)
    # LLM (see llm.PROVIDERS)
    "provider": "auto",  # auto | anthropic | claude-cli | openai
    "model": "",  # empty = provider default
    "base_url": "",  # OpenAI-compatible endpoint (or Anthropic proxy)
    "api_key": "",  # optional; otherwise read from api_key_env / provider default env var
    "api_key_env": "",
}


def load() -> dict:
    s = dict(DEFAULTS)
    try:
        s.update({k: v for k, v in json.loads(PATH.read_text()).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save(s: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(s, f, indent=2)
    os.chmod(PATH, 0o600)
