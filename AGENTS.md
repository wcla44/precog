# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, …) working in this repository.

## Commits and pull requests

- **Never add AI attribution of any kind.** No `Co-Authored-By: Claude …` (or any other AI co-author) trailer, and no "Generated with Claude Code" or similar line in commit messages, PR titles, or PR descriptions. Commits are authored by the human committer only.
- Only commit or push when the user asks you to.
- Keep commit messages short and imperative. Explain *why* in the body when it isn't obvious.

## Project

precog is a gaze-aware desktop assistant, written in Rust, for macOS and Linux (X11):
- webcam eye tracking (MediaPipe models converted to ONNX, run with `ort`)
- hand gestures: open hand = voice typing with local Whisper, fist = send, swipe left = cancel
- LLM suggestions based on a screenshot of the monitor you're looking at

`python/` contains the original Python prototype. It is kept as a reference; don't extend it.

## Build and test

```bash
scripts/convert_models.sh   # one-off: MediaPipe .task -> models/onnx/*.onnx (gitignored, needs uv)
scripts/fetch_unigaze.sh    # optional: UniGaze int8 ONNX (non-commercial weights: never commit them)
cargo build                 # dev profile is optimised (opt-level 2): the vision models are slow unoptimised
cargo test
./target/debug/precog --test-llm                  # screenshot -> LLM -> printed suggestion, no camera
./target/debug/precog --dump-landmarks img.jpg    # run the face/hand models on an image, print JSON
./target/debug/precog --capture X,Y,R             # copy what's under a halo there (moves the mouse!), print it
```

- The toolchain is pinned in `rust-toolchain.toml` (egui needs Rust ≥ 1.95).
- `.cargo/config.toml` sets `MACOSX_DEPLOYMENT_TARGET=14.4`, which the ONNX Runtime binaries require. `build.rs` lets two CoreML symbols resolve at runtime, for older Xcode SDKs.
- Linux build deps: `libgtk-3-dev libxdo-dev libasound2-dev cmake clang`; `xdotool` at runtime.

## Layout

| File | Role |
|---|---|
| `src/main.rs` | CLI, macOS camera permission prompt, eframe startup |
| `src/ui.rs` | egui app: HUD, gaze halo (smudge or shaped ring), multi-monitor and quick calibration, suggestion popup, voice overlay, settings window, gesture state machine |
| `src/vision.rs` | ONNX face mesh (478 points incl. iris), iris model, and two-hand landmarks with handedness: anchors, NMS, rotated ROI crops, tracking |
| `src/tracker.rs` | camera thread, gaze features, UniGaze and hands threads, swipe and head-shake detection, click polling |
| `src/camera_mac.rs` | native AVFoundation capture (nokhwa never delivers frames on macOS) |
| `src/geo.rs`, `src/layout.rs` | geometric gaze model: eyes and screens in 3D (camera frame), ray-screen intersection; physical screen arrangements (system layout + mm sizes) as starting points |
| `src/calib.rs` | two-phase calibration (laser dots with clicks, smooth pursuit with lag alignment and held-out error), quick recalibration |
| `src/gaze.rs` | ridge-regression gaze models (landmarks, UniGaze) with grouped CV, click corrector, fixation filter |
| `src/unigaze.rs` | UniGaze: PnP head pose, ETH-XGaze face normalisation, int8 ONNX inference |
| `src/gestures.rs` | pose classification (open / fist / pinch), hand openness, swipes, head shake, debounce |
| `src/halo.rs` | gaze halo: stays put inside, glides to new spots, left-hand resize and freeze |
| `src/capture.rs` | copy under the halo: X11 selection drag, OCR (tesseract), or image → clipboard |
| `src/xshape.rs` | Linux: compositor detection, halo window clipped to its ring (X11 SHAPE) |
| `src/dictation.rs` | microphone (cpal) → whisper.cpp → typed text |
| `src/agent.rs`, `src/llm.rs` | fixation trigger, screenshot + gaze trail, providers (Anthropic, Claude CLI, OpenAI-compatible) |
| `src/actions.rs` | executing suggested actions, focusing the window under the gaze |
| `src/tray.rs`, `src/settings.rs`, `src/paths.rs`, `src/monitors.rs` | tray icon, `~/.config/precog/config.json`, XDG paths, monitor layout |

## Conventions

- Coordinates are global logical screen coordinates (points on macOS, pixels on X11), with the origin at the top-left. This is the space used by window positions, mouse coordinates and Quartz window bounds.
- Any change to pre- or post-processing in `src/vision.rs` must be validated against MediaPipe on real images (`--dump-landmarks`). Face landmarks currently match within about 1px.
- Settings are stored at `~/.config/precog/config.json` with mode 600, because the file may contain an API key. Never log or commit secrets, model weights, or calibration recordings (`models/` is gitignored; recordings live in `~/.local/share/precog/`).
- Actions are always confirmed by the user (gaze plus gesture, or a click). Never add a code path that runs an LLM-suggested action without confirmation.
