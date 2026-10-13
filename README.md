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
scripts/fetch_unigaze.sh       # optional, recommended: UniGaze gaze model (~89 MB int8, non-commercial license)
cargo build --release
./target/release/precog
```

- The repo pins its own Rust toolchain in `rust-toolchain.toml`, which rustup installs automatically. `.cargo/config.toml` targets macOS 14.4 or later, because the ONNX Runtime binaries need it.
- Linux needs: `libgtk-3-dev libxdo-dev libasound2-dev libxcb*-dev cmake clang` and `xdotool` at runtime.
- **Linux with Nix** (no system packages needed; prefix each command with `nix-user-chroot ~/.nix` if you don't have root):
  - `nix develop -c cargo build --release` builds incrementally in a shell with every dependency.
  - `nix store add-path models/onnx --name precog-onnx-models`, run once after `convert_models.sh`, lets the next two commands embed the models.
  - `nix build` produces the wrapped `./result/bin/precog`.
  - `nix bundle` produces a single self-contained executable that runs on any x86_64 Linux. It includes Mesa for OpenGL and targets x86-64-v3 (AVX2) CPUs.
- Speech models are downloaded on first use into `~/.local/share/precog/whisper/`. The default is `base`, about 140 MB.

**macOS permissions** for your terminal app (System Settings → Privacy & Security):
- **Camera**
- **Microphone**
- **Screen Recording**: without it, screenshots show only your wallpaper.
- **Accessibility**: needed to type, focus windows and learn from clicks.

## Use

- **First run**: calibration in four steps, on every screen. 1) The mouse becomes a red laser: bring it onto each dot while following it with your eyes, then click the dot. 2) Reading: short texts appear in the corners of each screen; read them moving the laser under the words, and click at the end of each. 3) Follow a dot moving over all screens for ~18s; the red ring shows precog's guess and is refitted every second, so you can watch it converge (keep your eyes on the dot, not the ring). 4) Validation: look at fixed dots in the corners and centre of each screen; the error is measured per screen, centre vs corners, in px and degrees, and nothing is learned from it. On the test screen, Y finishes, R does one more pursuit (kept only if it does better on a new validation), A starts over. You don't have to sit still: leaning and shifting helps precog learn where you are. Screens are followed live: unplugging one, or moving it in the system settings, keeps the calibration (a screen is recognised by its resolution and physical size); a screen never seen before gets a short calibration of its own, the rest of the calibration is kept. Every calibration is recorded (even when cancelled) in `~/.local/share/precog/calibrations/` and can be replayed offline with `precog --replay FILE` to compare model variants.
- **Menu-bar icon 👁**: toggles for suggestions, gaze ring, hand gestures, voice typing and learning from clicks; plus Ask now, Recalibrate, Settings… and Quit.
- **Settings window**: everything else.
  - Fixation time and the minimum time between LLM calls.
  - Speech model and language.
  - "Click where I look" to focus a text field.
  - Inverting the swipe direction.
  - **AI provider and model**: Anthropic API, Claude Code CLI (uses your login), or any OpenAI-compatible endpoint (OpenAI, Ollama, OVHcloud AI Endpoints, OpenRouter, or a custom URL). Also the API key or the environment variable to read it from.
- **HUD** (bottom-left): fps, face detected, hand pose, gaze error, agent status. Click it to ask now; right-click to pause suggestions.

Right hand:

| Gesture | Effect |
|---|---|
| ✋ open hand (hold 0.5s) | Focus the window you're looking at and start voice typing. Text appears one phrase at a time. |
| ✊ closed fist | Finish and press Enter. |
| 👈 swipe left | Cancel voice typing and erase the text, dismiss a suggestion, or release a frozen halo. |
| 🤏 or ✊ while looking at ✓ | Accept a suggestion. |
| 🤏 or ✊ with the halo frozen | Copy what's under the halo to the clipboard (see below). |
| lower your hand + 5s of silence | Stop voice typing without pressing Enter. |

**Keyboard shortcuts** (global, whichever window has the focus), as a fallback for the gestures:

| Shortcut | Effect |
|---|---|
| Ctrl+Alt+H | Freeze / release the halo. |
| Ctrl+Alt+C | Copy what's under the halo (freezes it first). |
| Ctrl+Alt+V | Voice typing into the window you look at; again: send (Enter). |
| Ctrl+Alt+X | Cancel voice typing, or release the halo. |
| Ctrl+Alt+Q | Quit precog (also: the Quit button on the HUD, or the tray menu). |

Left and right hand are told apart by their side of your face (the hand on your right is your right hand). While a hand is in view, a **gesture panel** at the bottom of the main screen shows, for each hand, what precog sees (open, fist, pinch, raised or lowered) and what it will trigger in the current context, with a progress bar for the gestures you hold.

Left hand: the **gaze halo** stays still while your eyes move inside it, and glides to a new spot when you look elsewhere. Shake your head ("no") to reset it: released, back to its automatic size, back on your gaze. With a compositor (macOS, or picom on X11) it is the gaze smudge described below. On plain X11, which has no transparency, it is a red ring: the window is clipped to it with the SHAPE extension, so nothing else covers the screen.

| Gesture | Effect |
|---|---|
| open ✋ / close ✊ | Grow or shrink the halo. |
| raise it (fingers up) and hold it still for 1s | Freeze the halo (position and size). A hand resting on the desk doesn't freeze it. |

Copying under the frozen halo: in terminals, browsers and editors (Linux), precog drags a real selection across the zone and copies it. Elsewhere, or if nothing got selected, it runs OCR (`tesseract`, English and French) on a screenshot of the zone. If there is no text, it copies the screenshot as an image.

**Posture changes**: when you sit differently or tilt the laptop lid (which moves the camera), the HUD offers a quick recalibration (5 dots on one screen, 3 per screen otherwise, ~6s; also in the tray menu). Each dot is learned like a click you surely looked at, so the gaze is fixed at once for the new posture. Clicks keep refining it afterwards (see continuous recalibration below).

Command-line flags (they apply to this run only): `--calibrate`, `--no-agent`, `--no-gesture`, `--no-dot`, `--no-learn`, `--no-tray`, `--camera N`, `--test-llm` (one screenshot → LLM → print the suggestion).

## Files (XDG layout, also used on macOS)

- `~/.config/precog/config.json`: settings. Written with mode 600 because it may contain an API key. It is shared with the Python version.
- `~/.local/share/precog/gaze_model.json`: calibration and click samples, plus the screen layout they were recorded on. If the layout changes, precog recalibrates.
- `~/.local/share/precog/whisper/`: speech models.
- `~/.local/share/precog/calibrations/*.json`: raw face landmarks recorded during each calibration, for offline tuning of the gaze features and model.
- `~/.cache/precog/last_screen.jpg`: the last screenshot sent to the LLM. Only written when `PRECOG_DEBUG=1`.

**Gaze smudge**: with a compositor, the halo is a soft, animated haze. The centre is see-through, so you can still read what you're looking at. Until you size it with your left hand, its size reflects how noisy the gaze currently is.

**UniGaze** ([ut-vision/UniGaze](https://github.com/ut-vision/UniGaze), 2025) is an optional appearance-based gaze model, run with ONNX Runtime at about 10Hz. On a real two-screen calibration recording it cut the error from 269 to 156px per dot (162px with the int8 model), and it is much more robust to head movement. Its weights are non-commercial: `scripts/fetch_unigaze.sh` downloads and converts them locally, and they are never committed.

## How it works

- **Vision** (`src/vision.rs`): MediaPipe's face detector, 478-point face mesh (iris included), palm detector and hand landmark model, converted to ONNX and run with ONNX Runtime. Anchor decoding, rotated region-of-interest crops and frame-to-frame tracking follow MediaPipe. On test images, face landmarks are within about 1px of MediaPipe's.
- **Gaze features** (`src/tracker.rs`): for each eye, the iris centre is refined by MediaPipe's dedicated iris model (a 64×64 eye crop) and measured along the eye-corner axis, plus eyelid openness. Head yaw and pitch come from the face's 3D axes and from 2D geometry, plus roll and face position and scale. That's 14 values.
- **Eye patches** (`src/vision.rs`): as in WebGazer, a small grey image of each eye (10x6, aligned with the eye corners, contrast-normalised) is recorded with every frame. Its appearance (eyelid shape, how much of the iris shows) carries the vertical gaze better than the iris landmarks, where the gaze tends to be squeezed vertically. It is part of the regression: on a replayed calibration it cut the error on an unseen pursuit from 329 to 219px, and raised the vertical reach from 36% to 62% of the true range.
- **Geometric gaze model** (`src/geo.rs`): the head pose and eye position (mm) come from the face mesh (PnP on a metric 3D face model). The screens are placed in 3D in the camera frame: the laptop screen is just below its webcam, whatever the lid angle; the other screens start from the system layout and physical sizes (XRandR, Quartz), and their pose is fitted at calibration with Levenberg-Marquardt. The system layout is kept unless another arrangement explains the gaze clearly better. A per-user ridge regression maps eye features to the gaze direction in the head frame, and the gaze point is where that ray meets a screen. On a simulated user, at head positions not seen during calibration, this gives 38px median error against 349px for a direct pixel regression.
- **Gaze model** (`src/gaze.rs`): standardised features plus quadratic iris and head terms, fitted with ridge regression. The regularisation strength is picked by holding out whole calibration dots, so the error the HUD shows is honest. The output goes through a fixation-aware filter: samples are averaged while you're fixating, and the filter jumps as soon as 2–3 frames agree on a new spot (a saccade), so it doesn't lag. Every 5 clicks, precog refits using the eye data from 50–350ms before each click.
- **Gestures** (`src/gestures.rs`, `src/halo.rs`): both hands are tracked, each with its handedness; hands run on their own thread at the camera's frame rate. Open, fist and pinch are told apart from finger geometry; the left hand's openness sets the halo size. Swipes are detected from how fast and far the wrist moves, a head shake from the head's yaw going back and forth.
- **Voice typing** (`src/dictation.rs`): microphone audio goes through a simple voice-activity check that splits it into phrases at pauses; each phrase is transcribed by Whisper (Metal GPU on macOS) and typed.
- **Agent** (`src/agent.rs`): when your gaze stays within 160px for the fixation time, precog takes a screenshot of that monitor, draws your gaze trail on it, and sends it to the LLM. The LLM returns one JSON action or "none". Destructive commands are filtered out.
