# precog

See where you look, guess what you are about to do, and offer to do it. You confirm by looking at the suggestion and pinching 🤏.

```
webcam ─► MediaPipe face/iris/hand ─► gaze model (ridge regression) ─► gaze (x,y) ─► gaze dot
                                         ▲                                  │
              calibration dots + your mouse clicks (implicit learning)     ▼ fixation ≥1.8s
screen capture + gaze trail drawn on it ─────────────────────────────► Claude (vision) ─► suggestion popup
                                                                     look at ✓ for 1s + pinch ─► action
```

## Setup

```bash
uv sync
mkdir -p models && cd models && \
  curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task && \
  curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

LLM providers (choose from the tray menu → **AI**):
- **Auto** (default): uses the Anthropic API if `ANTHROPIC_API_KEY` is set, otherwise the Claude Code CLI.
- **Anthropic API**: Sonnet / Opus / Haiku 5.5.
- **Claude Code CLI**: uses your existing `claude` login. Slower, about 7–9s per call.
- **OpenAI-compatible**: OpenAI, Ollama (local), OVHcloud AI Endpoints, OpenRouter, or a custom URL. The model must accept images.

Model, endpoint and API key can be set from the menu (a small dialog opens) or in the config file.

**macOS permissions** (System Settings → Privacy & Security), all for your terminal app:
Camera, Screen Recording (without it the screenshot shows only your wallpaper), Accessibility + Input Monitoring (typing / click learning). Restart the terminal after granting them.

**Linux**: X11 recommended. Install `xdotool` (`apt install xdotool python3-tk`). On Wayland, screen capture and typing are restricted.

## Run

```bash
uv run python -m precog --no-agent      # 1) gaze only: calibrate, then check the red ring follows your eyes
uv run python -m precog                 # 2) full thing
uv run python -m precog --test-llm      # screenshot -> LLM -> print suggestion (no camera)
```

Options: `--calibrate` (redo), `--no-gesture` (confirm by looking at ✓ for 1.6s), `--no-dot`, `--no-learn`,
`--dwell 1.8` (how long you must fixate before the LLM is asked), `--interval 12` (minimum seconds between LLM calls).

**Menu-bar / tray icon** (👁; greyed out with a slash when suggestions are off). It shows the live status and has these items:
- Suggestions on/off
- Ask now
- Show gaze ring
- Pinch to confirm
- Learn from my clicks
- Recalibrate (train)
- Settings: fixation time, minimum time between asks
- AI: provider, model, endpoint, API key
- Quit

The icon runs as a separate process because tray libraries and Tk both need the main thread on macOS. Use `--no-tray` to disable it.
On Linux GNOME, the AppIndicator extension is needed to see tray icons.

HUD (bottom-left): left-click = ask now, right-click = pause the agent, shift-click = recalibrate. Ctrl-C in the terminal quits.

**Files** (XDG layout, also used on macOS):
- `~/.config/precog/config.json`: settings. Created with mode 600 because it may hold an API key. Command-line flags override it for one run only.
- `~/.local/share/precog/gaze_model.npz`: calibration and click samples.
- `~/.cache/precog/last_screen.jpg`: last screenshot sent to the LLM, only saved when `PRECOG_DEBUG=1`.

## How it works

- **Features** (`tracker.py`): iris position inside each eye (normalised by the eye-corner axis), eyelid openness, the 8 MediaPipe `eyeLook*` blendshapes, and head pose (pitch/yaw/roll + translation). That is 20 dimensions.
- **Model** (`model.py`): standardised features plus quadratic iris/head terms, fitted with ridge regression. Alpha is chosen by grouped k-fold, where held-out dots are never seen in training. The HUD shows that honest error (`err~Npx`). The output is smoothed with a One-Euro filter. The model is stored in `~/.local/share/precog/gaze_model.npz`.
- **Learning from clicks**: on every mouse click, the features from 50–350ms before the click are added as a sample at the cursor position, and the model is refitted every 5 clicks. It gets better (and adapts to drift) as you use your machine.
- **Agent** (`agent.py`): when your gaze stays within 160px for 1.8s, it takes a screenshot, draws your gaze trail on it, and sends it with the focused app name to Claude, which returns one JSON action or `none`. Destructive commands are filtered out.
- **Actions** (`actions.py`): `type_text` (into the app you were in, Enter only if `submit`), `run_command` (new terminal window), `open_url`, `open_app`, `hotkey`.

Expect about 2–4cm accuracy with a laptop webcam. That's enough to know which area or window you are looking at, but not which word.
