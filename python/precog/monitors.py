"""Monitor layout in global logical coordinates (points on macOS, pixels on X11), top-left origin.

mss reports exactly that on both platforms, and it is the same space Tk geometry, pynput and Quartz use.
"""
from __future__ import annotations

from dataclasses import dataclass

import mss


@dataclass(frozen=True)
class Monitor:
    index: int  # mss index (1-based)
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return min(max(x, self.left), self.right - 1), min(max(y, self.top), self.bottom - 1)

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.left}+{self.top}"


def list_monitors() -> list[Monitor]:
    with mss.mss() as sct:
        return [Monitor(i, m["left"], m["top"], m["width"], m["height"]) for i, m in enumerate(sct.monitors) if i > 0]


def monitor_at(monitors: list[Monitor], x: float, y: float) -> Monitor:
    for m in monitors:
        if m.contains(x, y):
            return m
    return min(monitors, key=lambda m: _dist2(m, x, y))


def clamp(monitors: list[Monitor], x: float, y: float) -> tuple[float, float]:
    """Snap a point that fell between/outside screens onto the nearest one."""
    return monitor_at(monitors, x, y).clamp(x, y)


def _dist2(m: Monitor, x: float, y: float) -> float:
    cx, cy = m.clamp(x, y)
    return (cx - x) ** 2 + (cy - y) ** 2
