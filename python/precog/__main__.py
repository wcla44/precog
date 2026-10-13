import argparse
import sys


def main():
    p = argparse.ArgumentParser(prog="precog", description="Gaze-aware proactive assistant")
    p.add_argument("--calibrate", action="store_true", help="force a new calibration")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-agent", action="store_true", help="gaze tracking only, no LLM")
    p.add_argument("--no-gesture", action="store_true", help="confirm by gaze dwell only (no pinch)")
    p.add_argument("--no-dot", action="store_true", help="hide the gaze dot")
    p.add_argument("--no-learn", action="store_true", help="don't learn from mouse clicks")
    p.add_argument("--no-tray", action="store_true", help="no menu-bar / tray icon")
    p.add_argument("--dwell", type=float, help="fixation seconds before asking the LLM (default 1.8)")
    p.add_argument("--interval", type=float, help="min seconds between LLM calls (default 12)")
    p.add_argument("--test-llm", action="store_true", help="one screenshot -> LLM -> print suggestion, then exit")
    args = p.parse_args()

    if args.test_llm:
        import json
        import time

        from . import actions
        from .agent import ask_llm, screenshot_with_gaze

        from .monitors import list_monitors

        mon = list_monitors()[0]
        cx, cy = mon.left + mon.width / 2, mon.top + mon.height / 2
        jpeg = screenshot_with_gaze([(time.monotonic(), cx, cy)], mon, time.monotonic())
        t0 = time.monotonic()
        from . import llm, settings

        cfg = settings.load()
        print("provider/model:", *llm.resolve(cfg))
        s = ask_llm(jpeg, actions.frontmost_app(), [], cfg)
        print(json.dumps(s, indent=2), f"\n({time.monotonic() - t0:.1f}s)")
        return

    from .app import App

    App(args).run()


if __name__ == "__main__":
    sys.exit(main())
