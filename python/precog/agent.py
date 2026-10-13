"""Intent agent: when the user's gaze settles, screenshot + gaze trail -> LLM -> one suggested action."""
from __future__ import annotations

import collections
import io
import json
import os
import queue
import re
import threading
import time

import mss
from PIL import Image, ImageDraw

from . import actions, llm
from .monitors import Monitor, monitor_at
from .paths import CACHE_DIR

SYSTEM = """You are Precog, a proactive desktop assistant. You see a screenshot of the user's screen.
The RED circle marks where the user is looking right now; YELLOW dots are where they looked during the last seconds
(older = smaller). Your job: anticipate the single next thing the user is about to do and offer to do it for them.

Only suggest something when you are fairly confident it is useful and specific (e.g. the user stares at an error
in a terminal -> suggest the fixing command; reads a URL / doc reference -> open it; looks at an empty Claude prompt
next to an error -> type a prompt asking Claude to fix it). Otherwise answer type "none". Being quiet is better than
being wrong. Never suggest destructive commands (rm -rf, sudo, force push, ...).

Action types:
- type_text: type `text` into the currently focused app ({app}); set `submit` true only if pressing Enter is clearly intended.
- run_command: open a new terminal and run shell `command`.
- open_url: open `url` in the browser.
- open_app: open application `app` by name.
- hotkey: press `keys`, e.g. ["cmd","t"].
- none: nothing worth suggesting.

Platform: {platform}. Focused app: {app}.
Recently suggested (avoid repeating unless the context clearly changed): {recent}

Reply with ONLY a JSON object:
{{"type": "...", "title": "<=8 words, what you will do", "reason": "<=20 words, why", "text": "", "submit": false,
  "command": "", "url": "", "app": "", "keys": []}}"""


def screenshot_with_gaze(trail: list[tuple[float, float, float]], mon: Monitor, now: float) -> bytes:
    """Capture the monitor the user looks at, with the gaze trail drawn in."""
    with mss.mss() as sct:
        raw = sct.grab({"left": mon.left, "top": mon.top, "width": mon.width, "height": mon.height})
    img = Image.frombytes("RGB", raw.size, raw.rgb)
    scale = img.width / mon.width  # retina: physical px per logical point
    draw = ImageDraw.Draw(img)

    def px(x, y):
        return (x - mon.left) * scale, (y - mon.top) * scale

    for t, x, y in trail[:-1]:
        r = max(3, 10 - (now - t) * 1.5) * scale
        cx, cy = px(x, y)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 210, 0))
    if trail:
        cx, cy = px(*trail[-1][1:])
        r = 45 * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=int(5 * scale))
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in LLM reply: {text[:200]}")
    return json.loads(m.group(0))


def ask_llm(jpeg: bytes, app: str, recent: list[str], cfg: dict | None = None) -> dict:
    import sys

    prompt = SYSTEM.format(app=app or "unknown", platform=sys.platform, recent=recent or "nothing")
    return _parse(llm.complete(cfg or {}, jpeg, prompt))


class Agent:
    """Watches the gaze stream; on a fixation, asks the LLM in a background thread."""

    def __init__(self, monitors: list[Monitor], dwell_s: float = 1.8, radius: float = 160, min_interval: float = 12.0):
        self.monitors = monitors
        self.dwell_s, self.radius, self.min_interval = dwell_s, radius, min_interval
        self.trail: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=400)
        self.suggestions: queue.Queue[dict] = queue.Queue()
        self.status = "idle"
        self.busy = False
        self.paused = False
        self.last_call = 0.0
        self.last_fix: tuple[float, float] | None = None
        self.recent: collections.deque[str] = collections.deque(maxlen=6)
        self.llm_cfg: dict = {}

    def update(self, t: float, x: float, y: float) -> None:
        self.trail.append((t, x, y))
        if self.busy or self.paused or t - self.last_call < self.min_interval:
            return
        pts = [(px, py) for pt, px, py in self.trail if t - pt <= self.dwell_s]
        if len(pts) < 10 or t - self.trail[0][0] < self.dwell_s:
            return
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if max(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 for px, py in pts) > self.radius:
            return  # still scanning, not fixating
        if self.last_fix and ((cx - self.last_fix[0]) ** 2 + (cy - self.last_fix[1]) ** 2) ** 0.5 < self.radius \
                and t - self.last_call < 4 * self.min_interval:
            return  # same spot as last time, don't spam
        self.trigger(t, (cx, cy))

    def trigger(self, t: float | None = None, fix: tuple[float, float] | None = None) -> None:
        t = t or time.monotonic()
        self.busy, self.last_call, self.last_fix = True, t, fix
        trail = [p for p in self.trail if t - p[0] <= 6.0]
        threading.Thread(target=self._run, args=(trail, t), daemon=True).start()

    def _run(self, trail, t) -> None:
        try:
            self.status = "thinking…"
            app = actions.frontmost_app()
            x, y = trail[-1][1:] if trail else (self.monitors[0].left, self.monitors[0].top)
            jpeg = screenshot_with_gaze(trail, monitor_at(self.monitors, x, y), t)
            if os.environ.get("PRECOG_DEBUG"):
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                (CACHE_DIR / "last_screen.jpg").write_bytes(jpeg)
            t0 = time.monotonic()
            s = ask_llm(jpeg, app, list(self.recent), self.llm_cfg)
            dt = time.monotonic() - t0
            if s.get("type") in actions.ACTION_TYPES and not actions.is_dangerous(s):
                s["_app"] = app
                self.recent.append(f"{s['type']}: {actions.describe(s)}")
                self.suggestions.put(s)
                self.status = f"suggested ({dt:.1f}s)"
            else:
                self.status = f"nothing to suggest ({dt:.1f}s)"
        except Exception as e:  # keep running whatever happens
            self.status = f"error: {e}"[:120]
        finally:
            self.busy = False
