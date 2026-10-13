"""Executes the actions the LLM may suggest. Cross-platform (macOS / Linux X11)."""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser

from pynput import keyboard

MAC = sys.platform == "darwin"

ACTION_TYPES = ["type_text", "run_command", "open_url", "open_app", "hotkey"]

DANGEROUS = re.compile(
    r"\brm\s+-[a-z]*r|\bsudo\b|\bmkfs|\bdd\s+if=|>\s*/dev/sd|\bgit\s+push\b.*(--force|-f)\b|"
    r"\bshutdown\b|\breboot\b|curl[^|]*\|\s*(ba|z)?sh|:\(\)\s*\{"
)


def is_dangerous(action: dict) -> bool:
    return bool(DANGEROUS.search(action.get("command", "") + " " + action.get("text", "")))


def frontmost_app() -> str:
    try:
        if MAC:
            return subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
        if shutil.which("xdotool"):
            return subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        pass
    return ""


def activate(app: str) -> None:
    if not app:
        return
    if MAC:
        subprocess.run(["osascript", "-e", f'tell application "{app}" to activate'], timeout=3)
    elif shutil.which("xdotool"):
        subprocess.run(["xdotool", "search", "--name", re.escape(app), "windowactivate"], timeout=3)
    time.sleep(0.35)


_KEYS = {
    "cmd": keyboard.Key.cmd, "command": keyboard.Key.cmd, "super": keyboard.Key.cmd,
    "ctrl": keyboard.Key.ctrl, "control": keyboard.Key.ctrl, "alt": keyboard.Key.alt,
    "option": keyboard.Key.alt, "shift": keyboard.Key.shift, "enter": keyboard.Key.enter,
    "return": keyboard.Key.enter, "tab": keyboard.Key.tab, "esc": keyboard.Key.esc,
    "escape": keyboard.Key.esc, "space": keyboard.Key.space, "up": keyboard.Key.up,
    "down": keyboard.Key.down, "left": keyboard.Key.left, "right": keyboard.Key.right,
    "backspace": keyboard.Key.backspace,
}


def execute(action: dict, target_app: str = "") -> None:
    kind = action["type"]
    kb = keyboard.Controller()
    if kind == "type_text":
        activate(target_app)
        kb.type(action["text"])
        if action.get("submit"):
            kb.tap(keyboard.Key.enter)
    elif kind == "run_command":
        cmd = action["command"]
        if MAC:
            esc = cmd.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(["osascript", "-e", f'tell application "Terminal" to do script "{esc}"',
                            "-e", 'tell application "Terminal" to activate'])
        else:
            for term in (["gnome-terminal", "--"], ["konsole", "-e"], ["x-terminal-emulator", "-e"], ["xterm", "-e"]):
                if shutil.which(term[0]):
                    subprocess.Popen(term + ["bash", "-c", f"{cmd}; exec bash"])
                    break
    elif kind == "open_url":
        webbrowser.open(action["url"])
    elif kind == "open_app":
        if MAC:
            subprocess.run(["open", "-a", action["app"]])
        else:
            subprocess.Popen(shlex.split(action["app"]))
    elif kind == "hotkey":
        activate(target_app)
        keys = [_KEYS.get(k.lower(), k.lower()) for k in action["keys"]]
        for k in keys:
            kb.press(k)
        for k in reversed(keys):
            kb.release(k)
    else:
        raise ValueError(f"unknown action {kind}")


def describe(action: dict) -> str:
    kind = action["type"]
    if kind == "type_text":
        return f"type: {action['text']}" + ("  ⏎" if action.get("submit") else "")
    if kind == "run_command":
        return f"$ {action['command']}"
    if kind == "open_url":
        return action["url"]
    if kind == "open_app":
        return f"open {action['app']}"
    if kind == "hotkey":
        return "+".join(action["keys"])
    return str(action)


# ---- focus the window the user is looking at -------------------------------------------------
def _mac_window_at(x: float, y: float) -> dict | None:
    import os

    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:  # front to back
        if w.get("kCGWindowLayer", 1) != 0 or w.get("kCGWindowOwnerPID") == os.getpid():
            continue
        b = w["kCGWindowBounds"]
        if b["X"] <= x < b["X"] + b["Width"] and b["Y"] <= y < b["Y"] + b["Height"]:
            return {"pid": w["kCGWindowOwnerPID"], "app": w.get("kCGWindowOwnerName", ""),
                    "title": w.get("kCGWindowName") or "", "bounds": (b["X"], b["Y"], b["Width"], b["Height"])}
    return None


def _mac_focus(win: dict) -> None:
    import AppKit

    app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(win["pid"])
    if app:
        app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
    try:  # raise that exact window (needs Accessibility permission)
        import ApplicationServices as AS

        ax = AS.AXUIElementCreateApplication(win["pid"])
        err, windows = AS.AXUIElementCopyAttributeValue(ax, "AXWindows", None)
        for w in windows or []:
            _, pos = AS.AXUIElementCopyAttributeValue(w, "AXPosition", None)
            ok, p = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
            if ok and abs(p.x - win["bounds"][0]) < 4 and abs(p.y - win["bounds"][1]) < 4:
                AS.AXUIElementPerformAction(w, "AXRaise")
                AS.AXUIElementSetAttributeValue(w, "AXMain", True)
                break
    except Exception:
        pass


def focus_at(x: float, y: float, click: bool = False) -> str:
    """Focus the window under (x, y); optionally click there to put the caret in the field.
    Returns a short description of the focused window."""
    from pynput import mouse

    m = mouse.Controller()
    if MAC:
        win = _mac_window_at(x, y)
        if win:
            _mac_focus(win)
        desc = f"{win['app']} — {win['title']}".strip(" —") if win else ""
    elif shutil.which("xdotool"):
        old = m.position
        subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y))], timeout=2)
        out = subprocess.run(["xdotool", "getmouselocation", "--shell"], capture_output=True, text=True, timeout=2).stdout
        wid = dict(line.split("=", 1) for line in out.split() if "=" in line).get("WINDOW")
        if wid:
            subprocess.run(["xdotool", "windowactivate", "--sync", wid], timeout=3)
        desc = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=2).stdout.strip()
        if not click:
            m.position = old
    else:
        desc = ""
    if click:
        old = m.position
        m.position = (x, y)
        time.sleep(0.05)
        m.click(mouse.Button.left)
        m.position = old
    time.sleep(0.15)
    return desc
