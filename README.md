# precog

precog watches where you look, guesses what you're about to do, and offers to do it for you. You confirm by looking at the suggestion and pinching 🤏.
It also does hands-free voice typing: open your hand ✋ and speak, and the text goes into the window you're looking at. Close your hand ✊ to send, or swipe left 👈 to cancel.

macOS and Linux (X11), written in Rust. The original Python prototype is in [`python/`](python/).

```
webcam ─► face/iris + hand models (MediaPipe → ONNX, ort) ─► gaze model (ridge) ─► gaze (x,y) on any monitor
                                     │                           ▲
                                     │        calibration dots + your mouse clicks (implicit learning)
                                     ▼
              ✋ open hand ─► focus window under gaze ─► mic ─► Whisper (local) ─► typed text   ✊ send · 👈 cancel

screenshot of the monitor you look at + gaze trail ─► LLM (vision) ─► suggestion popup ─► look at ✓ + 🤏/✊ ─► action
```

## Build

```bash
scripts/convert_models.sh      # one-off: MediaPipe .task -> models/onnx/*.onnx (needs uv)
cargo build --release
./target/release/precog
```

- The repo pins its own Rust toolchain in `rust-toolchain.toml`, which rustup installs automatically. `.cargo/config.toml` targets macOS 14.4 or later, because the ONNX Runtime binaries need it.
- Linux needs: `libgtk-3-dev libxdo-dev libasound2-dev libxcb*-dev cmake clang` and `xdotool` at runtime.
- Speech models are downloaded on first use into `~/.local/share/precog/whisper/`. The default is `base`, about 140 MB.

**macOS permissions** for your terminal app (System Settings → Privacy & Security):
- **Camera**
- **Microphone**
- **Screen Recording**: without it, screenshots show only your wallpaper.
- **Accessibility**: needed to type, focus windows and learn from clicks.

## Use

- **First run**: calibration runs on every screen. Click to start, then follow the dots with your eyes. Turn your head naturally when the dot changes screen.
- **Menu-bar icon 👁**: toggles for suggestions, gaze ring, hand gestures, voice typing and learning from clicks; plus Ask now, Recalibrate, Settings… and Quit.
- **Settings window**: everything else.
  - Fixation time and the minimum time between LLM calls.
  - Speech model and language.
  - "Click where I look" to focus a text field.
  - Inverting the swipe direction.
  - **AI provider and model**: Anthropic API, Claude Code CLI (uses your login), or any OpenAI-compatible endpoint (OpenAI, Ollama, OVHcloud AI Endpoints, OpenRouter, or a custom URL). Also the API key or the environment variable to read it from.
- **HUD** (bottom-left): fps, face detected, hand pose, gaze error, agent status. Click it to ask now; right-click to pause suggestions.

| Gesture | Effect |
|---|---|
| ✋ open hand (hold 0.3s) | Focus the window you're looking at and start voice typing. Text appears one phrase at a time. |
| ✊ closed fist | Finish and press Enter. |
| 👈 swipe left | Cancel voice typing and erase the text, or dismiss a suggestion. |
| 🤏 or ✊ while looking at ✓ | Accept a suggestion. |
| lower your hand + 5s of silence | Stop voice typing without pressing Enter. |

Command-line flags (they apply to this run only): `--calibrate`, `--no-agent`, `--no-gesture`, `--no-dot`, `--no-learn`, `--no-tray`, `--camera N`, `--test-llm` (one screenshot → LLM → print the suggestion).

## Files (XDG layout, also used on macOS)

- `~/.config/precog/config.json`: settings. Written with mode 600 because it may contain an API key. It is shared with the Python version.
- `~/.local/share/precog/gaze_model.json`: calibration and click samples, plus the screen layout they were recorded on. If the layout changes, precog recalibrates.
- `~/.local/share/precog/whisper/`: speech models.
- `~/.local/share/precog/calibrations/*.json`: raw face landmarks recorded during each calibration, for offline tuning of the gaze features and model.
- `~/.cache/precog/last_screen.jpg`: the last screenshot sent to the LLM. Only written when `PRECOG_DEBUG=1`.

**Gaze smudge**: where you look is shown as a soft, animated haze. The centre is see-through, so you can still read what you're looking at. Its size reflects how noisy the gaze currently is.

## How it works

- **Vision** (`src/vision.rs`): MediaPipe's face detector, 478-point face mesh (iris included), palm detector and hand landmark model, converted to ONNX and run with ONNX Runtime. Anchor decoding, rotated region-of-interest crops and frame-to-frame tracking follow MediaPipe. On test images, face landmarks are within about 1px of MediaPipe's.
- **Gaze features** (`src/tracker.rs`): for each eye, the iris centre is refined by MediaPipe's dedicated iris model (a 64×64 eye crop) and measured along the eye-corner axis, plus eyelid openness. Head yaw and pitch come from the face's 3D axes and from 2D geometry, plus roll and face position and scale. That's 14 values.
- **Gaze model** (`src/gaze.rs`): standardised features plus quadratic iris and head terms, fitted with ridge regression. The regularisation strength is picked by holding out whole calibration dots, so the error the HUD shows is honest. The output goes through a fixation-aware filter: samples are averaged while you're fixating, and the filter jumps as soon as 2–3 frames agree on a new spot (a saccade), so it doesn't lag. Every 5 clicks, precog refits using the eye data from 50–350ms before each click.
- **Gestures** (`src/gestures.rs`): open, fist and pinch are told apart from finger geometry. Swipes are detected from how fast and far the wrist moves.
- **Voice typing** (`src/dictation.rs`): microphone audio goes through a simple voice-activity check that splits it into phrases at pauses; each phrase is transcribed by Whisper (Metal GPU on macOS) and typed.
- **Agent** (`src/agent.rs`): when your gaze stays within 160px for the fixation time, precog takes a screenshot of that monitor, draws your gaze trail on it, and sends it to the LLM. The LLM returns one JSON action or "none". Destructive commands are filtered out.
