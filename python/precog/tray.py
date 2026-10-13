"""Menu-bar / system-tray icon, run as a child process of the app.

Tray libraries need the process main thread on macOS, and so does Tk, so the icon lives in its own
process. Protocol (JSON lines): app -> tray on stdin: {"status": str, "settings": {...}};
tray -> app on stdout: {"cmd": "set", "key": k, "value": v} | {"cmd": "ask"|"calibrate"|"quit"|"open_settings"}.
"""
from __future__ import annotations

import json
import sys
import threading

import pystray
from PIL import Image, ImageDraw

from .llm import ENDPOINTS, PROVIDERS, resolve

DWELLS = [1.0, 1.5, 1.8, 2.5, 3.5]
INTERVALS = [5.0, 12.0, 30.0, 60.0]
STT_MODELS = ["tiny", "base", "small", "large-v3-turbo"]
LANGS = [("", "Auto-detect"), ("en", "English"), ("fr", "Français"), ("de", "Deutsch"), ("es", "Español")]


def make_icon(active: bool) -> Image.Image:
    s = 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (255, 255, 255, 255) if active else (150, 150, 150, 255)
    d.ellipse([4, 16, s - 4, s - 16], outline=color, width=5)  # eye
    d.ellipse([22, 22, s - 22, s - 22], fill=color)  # pupil
    if not active:
        d.line([8, s - 8, s - 8, 8], fill=color, width=5)
    return img


def on_main(fn) -> None:
    if sys.platform == "darwin":
        from PyObjCTools import AppHelper

        AppHelper.callAfter(fn)
    else:
        fn()


def main() -> None:
    state = {"status": "starting…", "settings": json.loads(sys.argv[1])}
    cfg = state["settings"]

    def send(cmd: str, **kw) -> None:
        print(json.dumps({"cmd": cmd, **kw}), flush=True)

    def setter(key, value):
        def f(icon, item):
            cfg[key] = value
            send("set", key=key, value=value)
            refresh(icon)
        return f

    def toggle(key):
        def f(icon, item):
            setter(key, not cfg[key])(icon, item)
        return f

    def refresh(icon):
        icon.icon = make_icon(cfg["agent"])
        icon.update_menu()

    def checked(key):
        return lambda item: bool(cfg[key])

    def radio(key, values, fmt):
        return pystray.Menu(*[
            pystray.MenuItem(fmt(v), setter(key, v), checked=lambda item, v=v: cfg[key] == v, radio=True)
            for v in values
        ])

    def set_provider(name):
        def f(icon, item):
            setter("model", "")(icon, item)  # new provider -> its default model
            setter("provider", name)(icon, item)
        return f

    def set_endpoint(url):
        def f(icon, item):
            if cfg["provider"] != "openai":
                set_provider("openai")(icon, item)
            setter("base_url", url)(icon, item)
        return f

    def model_items():
        provider, model = resolve(cfg)
        presets = PROVIDERS[provider]["presets"]
        items = [pystray.MenuItem(m, setter("model", m), checked=lambda item, m=m: resolve(cfg)[1] == m, radio=True)
                 for m in presets]
        if model and model not in presets:
            items.append(pystray.MenuItem(model, None, checked=lambda item: True, radio=True))
        return items + [pystray.Menu.SEPARATOR, pystray.MenuItem("Custom model…", lambda icon, item: send("prompt", key="model"))]

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: state["status"], None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Suggestions on", toggle("agent"), checked=checked("agent")),
        pystray.MenuItem("Ask now", lambda icon, item: send("ask")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Show gaze ring", toggle("dot"), checked=checked("dot")),
        pystray.MenuItem("Hand gestures", toggle("gesture"), checked=checked("gesture")),
        pystray.MenuItem("Gestures & voice", pystray.Menu(
            pystray.MenuItem("✋ open hand → voice typing", toggle("voice"), checked=checked("voice")),
            pystray.MenuItem("Click where I look (focus the field)", toggle("click_focus"), checked=checked("click_focus")),
            pystray.MenuItem("Invert swipe direction", toggle("mirror"), checked=checked("mirror")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Speech model", radio("stt_model", STT_MODELS, lambda v: v)),
            pystray.MenuItem("Language", radio("stt_language", [c for c, _ in LANGS], lambda v: dict(LANGS)[v])),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("✊ fist: send · 👈 swipe left: cancel · 🤏/✊ on ✓: accept", None, enabled=False),
        )),
        pystray.MenuItem("Learn from my clicks", toggle("learn"), checked=checked("learn")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Recalibrate (train)…", lambda icon, item: send("calibrate")),
        pystray.MenuItem("Settings", pystray.Menu(
            pystray.MenuItem("Fixation before asking", radio("dwell", DWELLS, lambda v: f"{v:g} s")),
            pystray.MenuItem("Min time between asks", radio("interval", INTERVALS, lambda v: f"{v:g} s")),
            pystray.MenuItem("Open config file", lambda icon, item: send("open_settings")),
        )),
        pystray.MenuItem(lambda item: "AI: {} · {}".format(*resolve(cfg)), pystray.Menu(
            pystray.MenuItem("Provider", pystray.Menu(*[
                pystray.MenuItem(p["label"], set_provider(name), checked=lambda item, n=name: cfg["provider"] == n, radio=True)
                for name, p in PROVIDERS.items()
            ])),
            pystray.MenuItem("Model", pystray.Menu(model_items)),
            pystray.MenuItem("Endpoint (OpenAI-compatible)", pystray.Menu(*[
                pystray.MenuItem(f"{label}  {url}", set_endpoint(url), checked=lambda item, u=url: cfg["base_url"] == u, radio=True)
                for label, url in ENDPOINTS
            ], pystray.Menu.SEPARATOR, pystray.MenuItem("Custom URL…", lambda icon, item: send("prompt", key="base_url")))),
            pystray.MenuItem("Set API key…", lambda icon, item: send("prompt", key="api_key")),
            pystray.MenuItem("API key env variable…", lambda icon, item: send("prompt", key="api_key_env")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Test (ask now)", lambda icon, item: send("ask")),
        )),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit precog", lambda icon, item: (send("quit"), icon.stop())),
    )
    icon = pystray.Icon("precog", make_icon(cfg["agent"]), "precog", menu)

    def read_stdin():
        for line in sys.stdin:  # EOF = parent died -> exit
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            state["status"] = msg.get("status", state["status"])
            cfg.update(msg.get("settings", {}))
            on_main(lambda: refresh(icon))
        on_main(icon.stop)

    def setup(icon):
        icon.visible = True
        threading.Thread(target=read_stdin, daemon=True).start()

    if sys.platform == "darwin":  # menu-bar only, no Dock icon
        try:
            import AppKit

            AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
    icon.run(setup)


if __name__ == "__main__":
    main()
