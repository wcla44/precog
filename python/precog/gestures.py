"""Hand pose classification + swipe detection from MediaPipe hand landmarks (no LLM involved).

Poses: "open" (palm, fingers spread), "fist", "pinch" (thumb+index touching), "other".
Swipes are reported in the USER's frame of reference: "left" = the user moves their hand to their left.
"""
from __future__ import annotations

import collections

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]  # (mcp, pip, tip) index..pinky


def classify(lm: np.ndarray) -> str:
    """lm: (21, 2) normalised image coords."""
    size = float(np.linalg.norm(lm[WRIST] - lm[9])) or 1e-6
    extended = [
        np.linalg.norm(lm[tip] - lm[WRIST]) > np.linalg.norm(lm[pip] - lm[WRIST]) * 1.15
        for _, pip, tip in FINGERS
    ]
    curled = [np.linalg.norm(lm[tip] - lm[WRIST]) < np.linalg.norm(lm[mcp] - lm[WRIST]) * 1.1 for mcp, _, tip in FINGERS]
    thumb_index = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / size
    thumb_out = float(np.linalg.norm(lm[THUMB_TIP] - lm[5])) / size > 0.55
    if sum(curled) >= 4:
        return "fist"
    if thumb_index < 0.25 and sum(extended[1:]) >= 1:
        return "pinch"
    if all(extended) and thumb_out:
        return "open"
    return "other"


class SwipeDetector:
    WINDOW, MIN_DX, MAX_DY, COOLDOWN = 0.45, 0.30, 0.18, 1.0

    def __init__(self, mirrored: bool = False):
        # Raw (non-mirrored) webcam frames: the user's left is the image's right (+x).
        self.sign = -1 if mirrored else 1
        self.trail: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=30)
        self.last = -1e9

    def update(self, t: float, wrist: np.ndarray | None) -> str | None:
        if wrist is None:
            self.trail.clear()
            return None
        self.trail.append((t, float(wrist[0]), float(wrist[1])))
        while self.trail and t - self.trail[0][0] > self.WINDOW:
            self.trail.popleft()
        if t - self.last < self.COOLDOWN or len(self.trail) < 3:
            return None
        _, x0, y0 = self.trail[0]
        dx, dy = (self.trail[-1][1] - x0) * self.sign, self.trail[-1][2] - y0
        if abs(dx) > self.MIN_DX and abs(dy) < self.MAX_DY:
            self.last = t
            self.trail.clear()
            return "left" if dx > 0 else "right"
        return None


class Debounce:
    """A pose only counts once it has been held for `hold` seconds."""

    def __init__(self, hold: float = 0.3):
        self.hold = hold
        self.pose: str | None = None
        self.since = 0.0
        self.stable: str | None = None

    def update(self, t: float, pose: str | None) -> str | None:
        if pose != self.pose:
            self.pose, self.since = pose, t
        if t - self.since >= self.hold:
            self.stable = pose
        return self.stable
