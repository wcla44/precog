"""Tk UI: multi-monitor calibration, gaze dot, HUD, suggestion popup, voice-typing overlay, hand gestures."""
from __future__ import annotations

import json
import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk

import numpy as np

from . import actions
from . import settings as settings_mod
from .agent import Agent
from .dictation import Dictation
from .gestures import Debounce
from .model import MODEL_PATH, GazeModel, OneEuro
from .monitors import Monitor, clamp, list_monitors, monitor_at
from .tracker import Tracker

MAC = sys.platform == "darwin"
TICK_MS = 33
POSE_ICON = {"open": "✋", "fist": "✊", "pinch": "🤏", "other": "🖐", None: "  "}

PROMPTS = {  # key -> dialog title, question, secret
    "model": ("Model", "Model name (empty = provider default),\ne.g. claude-sonnet-5-5, opus, qwen2.5vl:7b:", False),
    "base_url": ("Endpoint", "OpenAI-compatible base URL (…/v1):", False),
    "api_key": ("API key", "API key (stored in ~/.config/precog/config.json, mode 600).\nEmpty = use the env variable:", True),
    "api_key_env": ("API key env variable", "Environment variable holding the API key (e.g. OPENAI_API_KEY):", False),
}


def _layout(monitors: list[Monitor]) -> list[tuple[int, int, int, int]]:
    return [(m.left, m.top, m.width, m.height) for m in monitors]


class Calibration:
    """One black window per monitor; dots are shown screen by screen, so the model learns the whole desktop."""

    GRID = 4
    SETTLE, COLLECT, SWITCH_SETTLE = 0.7, 1.0, 1.4

    def __init__(self, app: "App", on_done):
        self.app, self.on_done = app, on_done
        self.screens = []
        for m in app.monitors:
            top = tk.Toplevel(app.root, bg="black")
            top.overrideredirect(True)
            top.attributes("-topmost", True)
            top.geometry(m.geometry)
            c = tk.Canvas(top, bg="black", highlightthickness=0, width=m.width, height=m.height)
            c.pack(fill="both", expand=True)
            c.create_text(m.width / 2, m.height / 2, fill="white", font=("Helvetica", 26), justify="center",
                          tags="intro",
                          text=f"Calibration · {len(app.monitors)} screen(s)\n\n"
                               "Follow each dot with your EYES. Turn your head naturally when it changes screen.\n"
                               "Sit the way you usually work.\n\nCLICK anywhere to start · ESC to cancel")
            for w in (top, c):
                w.bind("<Button-1>", lambda e: self._next() if self.i < 0 else None)
                w.bind("<Escape>", lambda e: self._finish(abort=True))
            self.screens.append((m, top, c))
        ticks = np.linspace(0.05, 0.95, self.GRID)
        self.points: list[tuple[int, float, float]] = []
        for si, (m, _, _) in enumerate(self.screens):
            pts = [(si, m.left + float(x) * m.width, m.top + float(y) * m.height) for x in ticks for y in ticks]
            random.shuffle(pts)
            self.points += pts
        self.model = GazeModel()
        self.model.layout = _layout(app.monitors)
        self.i = -1
        self.screens[0][1].after(300, self.screens[0][1].focus_force)

    def _next(self):
        if self.i < 0:
            for _, top, c in self.screens:
                c.delete("intro")
                top.focus_force()
        self.i += 1
        if self.i >= len(self.points):
            return self._finish()
        switched = self.i > 0 and self.points[self.i][0] != self.points[self.i - 1][0]
        self.settle = self.SWITCH_SETTLE if switched or self.i == 0 else self.SETTLE
        self.t0 = time.monotonic()
        self._animate()

    def _animate(self):
        si, x, y = self.points[self.i]
        m, top, c = self.screens[si]
        el = time.monotonic() - self.t0
        r = max(6, 34 * (1 - el / self.settle)) if el < self.settle else 6
        for _, _, cc in self.screens:
            cc.delete("dot")
        lx, ly = x - m.left, y - m.top
        c.create_text(m.width / 2, 30, fill="#666", text=f"{self.i + 1}/{len(self.points)}", tags="dot")
        c.create_oval(lx - r, ly - r, lx + r, ly + r, fill="#ff3b30", outline="", tags="dot")
        c.create_oval(lx - 2, ly - 2, lx + 2, ly + 2, fill="white", outline="", tags="dot")
        if el < self.settle + self.COLLECT:
            top.after(16, self._animate)
        else:
            frames = self.app.tracker.frames_between(self.t0 + self.settle, self.t0 + self.settle + self.COLLECT)
            if len(frames) >= 5:
                self.model.add(np.array([f.features for f in frames]), (x, y), group=self.i)
            self._next()

    def _finish(self, abort=False):
        for _, top, _ in self.screens:
            top.destroy()
        self.screens = []
        if abort or len(self.model.F) < 30:
            return self.on_done(None)
        self.model.fit()
        self.model.save()
        self.on_done(self.model)


class GazeDot:
    S = 44

    def __init__(self, root):
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        if MAC:
            self.top.attributes("-transparent", True)
            bg = "systemTransparent"
        else:
            self.top.attributes("-alpha", 0.55)
            bg = "black"
        self.top.config(bg=bg)
        c = tk.Canvas(self.top, width=self.S, height=self.S, bg=bg, highlightthickness=0)
        c.pack()
        c.create_oval(4, 4, self.S - 4, self.S - 4, outline="#ff3b30", width=4)
        self.hidden = False

    def move(self, x, y, mouse):
        near = (x - mouse[0]) ** 2 + (y - mouse[1]) ** 2 < 80 ** 2  # never sit under the cursor (would eat clicks)
        if near != self.hidden:
            (self.top.withdraw if near else self.top.deiconify)()
            self.hidden = near
        if not near:
            self.top.geometry(f"+{int(x - self.S / 2)}+{int(y - self.S / 2)}")

    def hide(self):
        if not self.hidden:
            self.top.withdraw()
            self.hidden = True


class Popup:
    W, H = 600, 270
    ACCEPT_DWELL, DISMISS_DWELL, TIMEOUT = 1.0, 1.3, 30.0

    def __init__(self, app: "App", s: dict, mon: Monitor):
        self.app, self.s = app, s
        self.result: str | None = None
        self.top = top = tk.Toplevel(app.root, bg="#1c1c1e")
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        self.x0, self.y0 = mon.right - self.W - 24, mon.top + 48
        top.geometry(f"{self.W}x{self.H}+{self.x0}+{self.y0}")
        tk.Label(top, text="🔮 " + s.get("title", "Suggestion"), fg="white", bg="#1c1c1e",
                 font=("Helvetica", 18, "bold"), anchor="w").pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(top, text=s.get("reason", ""), fg="#aaa", bg="#1c1c1e", font=("Helvetica", 12),
                 anchor="w", wraplength=self.W - 32, justify="left").pack(fill="x", padx=16)
        tk.Label(top, text=actions.describe(s)[:300], fg="#7ee787", bg="#0d1117", font=("Menlo", 12),
                 anchor="w", wraplength=self.W - 48, justify="left").pack(fill="x", padx=16, pady=8, ipady=6, ipadx=6)
        row = tk.Frame(top, bg="#1c1c1e")
        row.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.zones = {}
        for name, color in (("accept", "#1f6f3f"), ("dismiss", "#5a1f1f")):
            c = tk.Canvas(row, bg=color, highlightthickness=0, height=100)
            c.pack(side="left", fill="both", expand=True, padx=4)
            c.bind("<Button-1>", lambda e, n=name: self.done(n))
            self.zones[name] = c
        self.dwell = {"accept": 0.0, "dismiss": 0.0}
        self.created = time.monotonic()

    def _zone_at(self, gx, gy):
        m = 90  # gaze is ~2-3cm accurate: be generous around the popup
        if not (self.x0 - m <= gx <= self.x0 + self.W + m and self.y0 - m <= gy <= self.y0 + self.H + m):
            return None
        return "accept" if gx < self.x0 + self.W / 2 else "dismiss"

    def update(self, gaze, confirm: bool, gestures: bool, dt: float):
        """confirm: a confirming hand pose (pinch or closed fist) is held right now."""
        if self.result:
            return
        zone = self._zone_at(*gaze) if gaze else None
        for n in self.dwell:
            self.dwell[n] = self.dwell[n] + dt if n == zone else max(0.0, self.dwell[n] - 2 * dt)
        need = self.ACCEPT_DWELL if gestures else 1.6
        a = self.dwell["accept"] >= need
        if a and (confirm or not gestures):
            return self.done("accept")
        if self.dwell["dismiss"] >= self.DISMISS_DWELL or time.monotonic() - self.created > self.TIMEOUT:
            return self.done("dismiss")
        for n, c in self.zones.items():
            c.delete("all")
            w, h = c.winfo_width(), c.winfo_height()
            lim = need if n == "accept" else self.DISMISS_DWELL
            frac = min(1.0, self.dwell[n] / lim)
            c.create_rectangle(0, h - 8, w * frac, h, fill="white", outline="")
            if n == "accept":
                label = ("✓ ACCEPT\n🤏/✊ now!" if a else "✓ ACCEPT\nlook + pinch 🤏 or ✊") if gestures else "✓ ACCEPT\nlook here"
            else:
                label = "✗ DISMISS\nlook here · or swipe 👈" if gestures else "✗ DISMISS\nlook here"
            c.create_text(w / 2, h / 2 - 4, text=label, fill="white", font=("Helvetica", 17, "bold"), justify="center")

    def done(self, result):
        if self.result:
            return
        self.result = result
        self.top.destroy()


class VoiceOverlay:
    W = 680

    def __init__(self, root):
        self.top = tk.Toplevel(root, bg="#1c1c1e")
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.head = tk.Label(self.top, fg="white", bg="#1c1c1e", font=("Helvetica", 15, "bold"), anchor="w")
        self.head.pack(fill="x", padx=14, pady=(10, 2))
        self.text = tk.Label(self.top, fg="#7ee787", bg="#0d1117", font=("Menlo", 13), anchor="w", justify="left",
                             wraplength=self.W - 40)
        self.text.pack(fill="x", padx=14, pady=4, ipady=6, ipadx=6)
        tk.Label(self.top, text="✊ close hand: send ⏎   ·   👈 swipe left: cancel   ·   lower hand + pause: stop",
                 fg="#999", bg="#1c1c1e", font=("Helvetica", 11)).pack(padx=14, pady=(0, 10))
        self.top.withdraw()
        self.hide_at: float | None = None

    def show(self, mon: Monitor, target: str):
        self.target = target
        self.head.config(text=f"🎙  Listening → {target or 'window under your gaze'}")
        self.text.config(text="…")
        self.top.geometry(f"{self.W}x130+{mon.left + (mon.width - self.W) // 2}+{mon.bottom - 190}")
        self.top.deiconify()
        self.hide_at = None

    def status(self, state: str, text: str):
        icons = {"listening": "🎙  Listening", "finishing": "⏳ Transcribing", "sent": "✅ Sent",
                 "done": "✅ Typed", "cancelled": "↩️  Cancelled (erased)", "error": "⚠️  Error"}
        if state in icons:
            self.head.config(text=f"{icons[state]} → {self.target or ''}".rstrip(" →"))
        if text:
            self.text.config(text=text[-300:])
        if state in ("sent", "done", "cancelled", "error"):
            self.hide_at = time.monotonic() + (4.0 if state == "error" else 1.5)

    def tick(self, now):
        if self.hide_at and now > self.hide_at:
            self.top.withdraw()
            self.hide_at = None


class App:
    def __init__(self, args):
        self.args = args
        self.saved = settings_mod.load()
        self.cfg = dict(self.saved)
        # CLI flags override saved settings for this session only
        for flag, key in (("no_agent", "agent"), ("no_gesture", "gesture"), ("no_dot", "dot"), ("no_learn", "learn")):
            if getattr(args, flag):
                self.cfg[key] = False
        for key in ("dwell", "interval"):
            if getattr(args, key) is not None:
                self.cfg[key] = getattr(args, key)
        self.root = tk.Tk()
        self.root.withdraw()
        self.monitors = list_monitors()
        self.primary = self.monitors[0]
        self.tracker = Tracker(camera=args.camera, hands=self.cfg["gesture"]).start()
        self.model: GazeModel | None = None
        if MODEL_PATH.exists() and not args.calibrate:
            try:
                self.model = GazeModel.load()
                if self.model.layout != _layout(self.monitors):
                    print("screen layout changed since last calibration: recalibrating")
                    self.model = None
            except Exception as e:
                print(f"could not load model ({e}), recalibrating")
                self.model = None
        self.filter = OneEuro()
        self.gaze: tuple[float, float] | None = None
        self.dot = GazeDot(self.root)
        self.agent = Agent(self.monitors)
        self.popup: Popup | None = None
        self.voice_events: queue.Queue = queue.Queue()
        self.dictation = Dictation(lambda state, text: self.voice_events.put((state, text)))
        self.overlay = VoiceOverlay(self.root)
        self.voice_starting = False
        self.voice_cooldown = 0.0
        self.pose = Debounce(0.3)
        self.prev_pose: str | None = None
        self.clicks: queue.Queue = queue.Queue()
        self.n_clicks = 0
        self.last_tick = time.monotonic()
        self._make_hud()
        self._start_click_listener()
        self.cmds: queue.Queue = queue.Queue()
        self.tray = None if args.no_tray else self._start_tray()
        self.tray_status = ""
        self._apply()
        if self.cfg["gesture"] and self.cfg["voice"]:
            self.dictation.preload()
        self.calibrating = False
        self.root.after(800, self._maybe_calibrate)
        self.root.after(TICK_MS, self.tick)

    # ---- setup -------------------------------------------------------------
    def _make_hud(self):
        self.hud = tk.Toplevel(self.root, bg="#111")
        self.hud.overrideredirect(True)
        self.hud.attributes("-topmost", True)
        if not MAC:
            self.hud.attributes("-alpha", 0.85)
        self.hud.geometry(f"+{self.primary.left + 12}+{self.primary.bottom - 70}")
        self.hud_label = tk.Label(self.hud, fg="#ddd", bg="#111", font=("Menlo", 11), justify="left", padx=8, pady=4)
        self.hud_label.pack()
        tip = "left-click: ask now · right-click: pause agent · shift-click: recalibrate"
        tk.Label(self.hud, text=tip, fg="#777", bg="#111", font=("Menlo", 9), padx=8).pack()
        for w in (self.hud, self.hud_label):
            w.bind("<Button-1>", self._hud_click)
            w.bind("<Shift-Button-1>", lambda e: self._calibrate())
            w.bind("<Button-2>" if MAC else "<Button-3>", self._hud_pause)
            w.bind("<Button-3>", self._hud_pause)

    def _hud_click(self, e):
        if e.state & 0x1:  # shift
            return
        if not self.agent.busy:
            self.agent.trigger(fix=self.gaze)

    def _hud_pause(self, e):
        self._set("agent", not self.cfg["agent"])
        self._tray_send({"settings": {"agent": self.cfg["agent"]}})

    # ---- settings / tray ---------------------------------------------------
    def _apply(self):
        c = self.cfg
        self.agent.paused = not c["agent"]
        self.agent.dwell_s, self.agent.min_interval = c["dwell"], c["interval"]
        self.agent.llm_cfg = {k: c[k] for k in ("provider", "model", "base_url", "api_key", "api_key_env")}
        self.tracker.use_hands = c["gesture"]
        self.tracker.mirrored = c["mirror"]
        self.dictation.configure(c["stt_model"], c["stt_language"])

    def _set(self, key, value):
        self.cfg[key] = self.saved[key] = value
        settings_mod.save(self.saved)
        self._apply()
        if key == "stt_model" or (key in ("voice", "gesture") and value):
            self.dictation.preload()

    def _prompt(self, key):
        """Free-text setting (model name, endpoint URL, API key) via a small Tk dialog."""
        from tkinter import simpledialog

        title, question, secret = PROMPTS[key]
        holder = tk.Toplevel(self.root)  # visible, topmost parent so the dialog isn't hidden behind other apps
        holder.overrideredirect(True)
        holder.attributes("-topmost", True)
        p = self.primary
        holder.geometry(f"1x1+{p.left + p.width // 2}+{p.top + p.height // 3}")
        if MAC:
            import os

            subprocess.run(["osascript", "-e", "tell application \"System Events\" to set frontmost of "
                            f"(first process whose unix id is {os.getpid()}) to true"], capture_output=True)
        holder.lift()
        holder.focus_force()
        value = simpledialog.askstring(title, question, parent=holder, show="•" if secret else None,
                                       initialvalue="" if secret else self.cfg[key])
        holder.destroy()
        if value is None:
            return
        self._set(key, value.strip())
        self._tray_send({"settings": {key: self.cfg[key]}})

    def _start_tray(self):
        try:
            p = subprocess.Popen([sys.executable, "-m", "precog.tray", json.dumps(self.cfg)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            print("tray icon unavailable:", e)
            return None

        def read():
            for line in p.stdout:
                try:
                    self.cmds.put(json.loads(line))
                except ValueError:
                    pass

        threading.Thread(target=read, daemon=True).start()
        return p

    def _tray_send(self, msg: dict):
        if self.tray and self.tray.poll() is None:
            try:
                self.tray.stdin.write(json.dumps(msg) + "\n")
            except OSError:
                pass

    def _handle_cmds(self):
        while True:
            try:
                m = self.cmds.get_nowait()
            except queue.Empty:
                return
            cmd = m.get("cmd")
            if cmd == "set" and m.get("key") in self.cfg:
                self._set(m["key"], m["value"])
            elif cmd == "ask" and not self.agent.busy:
                self.agent.trigger(fix=self.gaze)
            elif cmd == "prompt" and m.get("key") in PROMPTS:
                self._prompt(m["key"])
            elif cmd == "calibrate":
                self._calibrate()
            elif cmd == "open_settings":
                settings_mod.save(self.saved)
                subprocess.Popen(["open" if MAC else "xdg-open", str(settings_mod.PATH)])
            elif cmd == "quit":
                self.root.quit()

    def _start_click_listener(self):
        from pynput import mouse

        def on_click(x, y, button, pressed):
            if pressed and self.cfg["learn"]:
                self.clicks.put((time.monotonic(), x, y))

        mouse.Listener(on_click=on_click, daemon=True).start()

    def _maybe_calibrate(self):
        if self.model is None:
            self._calibrate()

    def _calibrate(self):
        if self.calibrating:
            return
        self.calibrating = True
        self.hud.withdraw()

        def done(model):
            self.calibrating = False
            self.hud.deiconify()
            if model is not None:
                self.model = model
                self.n_clicks = 0
                print(f"calibrated on {len(self.monitors)} screen(s): {len(model.F)} samples, "
                      f"cross-validated median error {model.cv_error:.0f}px")
            elif self.model is None:
                print("calibration cancelled/failed (is your face visible?)")

        Calibration(self, done)

    # ---- main loop ---------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        dt, self.last_tick = now - self.last_tick, now
        try:
            self._tick(now, dt)
        except Exception as e:
            print("tick error:", e)
        self.root.after(TICK_MS, self.tick)

    def _tick(self, now, dt):
        f = self.tracker.latest
        if self.tracker.error:
            print(self.tracker.error)
            self.root.quit()
            return
        face = f is not None and f.features is not None and now - f.t < 0.3
        if face and self.model and self.model.ready and not self.calibrating:
            x, y = clamp(self.monitors, *self.model.predict(f.features))
            x, y = self.filter(np.array([x, y]), now)
            self.gaze = (float(x), float(y))
        elif not face:
            self.gaze = None

        self._learn_from_clicks()
        self._handle_cmds()

        if self.gaze and not self.calibrating and self.cfg["dot"]:
            self.dot.move(*self.gaze, self.root.winfo_pointerxy())
        else:
            self.dot.hide()

        pose = self._gestures(now, f)

        if not self.calibrating:
            if self.popup:
                confirm = pose in ("pinch", "fist")
                self.popup.update(self.gaze, confirm, self.cfg["gesture"], dt)
                if self.popup.result:
                    self._resolve(self.popup)
                    self.popup = None
            elif self.dictation.state == "idle" and not self.voice_starting:
                if self.gaze:
                    self.agent.update(now, *self.gaze)
                try:
                    s = self.agent.suggestions.get_nowait()
                    g = self.gaze or (self.primary.left, self.primary.top)
                    self.popup = Popup(self, s, monitor_at(self.monitors, *g))
                except queue.Empty:
                    pass

        while True:
            try:
                self.overlay.status(*self.voice_events.get_nowait())
            except queue.Empty:
                break
        self.overlay.tick(now)
        self._update_hud(f, face, pose)

    # ---- gestures (no LLM) ---------------------------------------------------
    def _gestures(self, now, f) -> str | None:
        swipes = []
        while True:
            try:
                swipes.append(self.tracker.swipes.get_nowait()[1])
            except queue.Empty:
                break
        if not self.cfg["gesture"] or self.calibrating:
            return None
        pose = self.pose.update(now, f.pose if f is not None and now - f.t < 0.3 else None)
        rising = pose if pose != self.prev_pose else None
        self.prev_pose = pose

        if self.popup:
            if "left" in swipes:
                self.popup.done("dismiss")
            return pose
        if not self.cfg["voice"]:
            return pose
        state = self.dictation.state
        if state == "idle" and not self.voice_starting:
            if rising == "open" and self.gaze and now > self.voice_cooldown:
                self._start_voice(*self.gaze)
        elif state == "listening":
            if "left" in swipes:
                self.dictation.cancel()
                self.voice_cooldown = now + 1.5
            elif rising == "fist":
                self.dictation.send()
                self.voice_cooldown = now + 1.5
            elif pose is None and self.dictation.silent_for() > 5:
                self.dictation.stop()
                self.voice_cooldown = now + 1.0
        return pose

    def _start_voice(self, gx, gy):
        self.voice_starting = True
        mon = monitor_at(self.monitors, gx, gy)
        self.overlay.show(mon, "")  # show first: our window must not grab focus after the target got it

        def run():
            try:
                time.sleep(0.1)
                target = actions.focus_at(gx, gy, click=self.cfg["click_focus"])
                self.overlay.target = target
                self.dictation.start()
            except Exception as e:
                self.voice_events.put(("error", f"voice typing failed: {e}"))
            finally:
                self.voice_starting = False

        threading.Thread(target=run, daemon=True).start()

    # ---- suggestions -------------------------------------------------------
    def _resolve(self, popup: Popup):
        s = popup.s
        print(f"[{popup.result}] {s.get('title')} — {actions.describe(s)}")
        self.agent.last_call = time.monotonic()  # cool down after a decision
        if popup.result == "accept":
            threading.Thread(target=self._exec, args=(s,), daemon=True).start()

    @staticmethod
    def _exec(s):
        try:
            actions.execute(s, s.get("_app", ""))
        except Exception as e:
            print("action failed:", e)

    def _learn_from_clicks(self):
        """Implicit calibration: people look at what they click ~100-300ms before the click."""
        refit = False
        while True:
            try:
                t, x, y = self.clicks.get_nowait()
            except queue.Empty:
                break
            if not (self.model and self.model.ready) or self.calibrating or self.popup:
                continue
            frames = self.tracker.frames_between(t - 0.35, t - 0.05)
            if len(frames) < 3:
                continue
            self.model.add(np.array([fr.features for fr in frames]), (x, y), group=1000 + self.n_clicks, weight=0.5)
            self.n_clicks += 1
            refit = refit or self.n_clicks % 5 == 0
        if refit:
            self.model.fit()
            self.model.save()

    def _update_hud(self, f, face, pose):
        parts = [f"{self.tracker.fps:4.1f}fps", "face ✓" if face else "face ✗"]
        if self.cfg["gesture"]:
            parts.append(POSE_ICON.get(pose, "  "))
        if self.model and self.model.ready:
            parts.append(f"err~{self.model.cv_error:.0f}px clicks:{self.n_clicks}")
        if self.dictation.state != "idle":
            parts.append("🎙 " + self.dictation.state)
        parts.append("agent: " + ("paused" if self.agent.paused else self.agent.status))
        self.hud_label.config(text="precog · " + " · ".join(parts))
        status = ("● " if face else "○ ") + ("paused" if self.agent.paused else self.agent.status)
        if self.dictation.state != "idle":
            status = "🎙 " + self.dictation.state
        if self.calibrating:
            status = "calibrating…"
        if status != self.tray_status:
            self.tray_status = status
            self._tray_send({"status": status})

    def run(self):
        import signal

        signal.signal(signal.SIGINT, lambda *a: self.root.quit())
        self.root.mainloop()
        self.tracker.stop()
        if self.tray:
            self.tray.terminate()
