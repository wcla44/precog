"""Voice typing: microphone -> local Whisper (faster-whisper) -> keystrokes into the focused field.

Text is typed incrementally, one phrase at a time (cut on pauses), so it appears while you speak.
send() finishes and presses Enter, stop() finishes without Enter, cancel() erases what was typed.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

SR = 16000
FRAME = 480  # 30 ms
PAUSE_S = 0.7  # silence that ends a phrase
MAX_PHRASE_S = 12.0


class Dictation:
    def __init__(self, on_status: Callable[[str, str], None]):
        self.on_status = on_status  # (state, text) -> None, called from the worker thread
        self.model_size = "base"
        self.language: str | None = None
        self.state = "idle"  # idle | listening | finishing
        self._model = None
        self._model_key = None
        self._lock = threading.Lock()
        self._buf: list[np.ndarray] = []
        self._stream = None
        self._typed = 0
        self._text = ""
        self._last_voice = 0.0
        self._cmds: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="dictation").start()

    # ---- public API (any thread) ------------------------------------------
    def configure(self, model_size: str, language: str) -> None:
        self.model_size, self.language = model_size, (language or None)

    def preload(self) -> None:
        self._cmds.put("preload")

    def start(self) -> None:
        if self.state != "idle":
            return
        import sounddevice as sd

        with self._lock:
            self._buf, self._typed, self._text = [], 0, ""
        self._last_voice = time.monotonic()
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                      callback=self._audio)
        self._stream.start()
        self.state = "listening"
        self.on_status("listening", "")

    def send(self) -> None:
        self._cmds.put("send")

    def stop(self) -> None:
        self._cmds.put("stop")

    def cancel(self) -> None:
        self._cmds.put("cancel")

    def silent_for(self) -> float:
        return time.monotonic() - self._last_voice

    # ---- internals ----------------------------------------------------------
    def _audio(self, data, frames, t, status):
        with self._lock:
            self._buf.append(data[:, 0].copy())

    def _load(self):
        key = self.model_size
        if self._model_key != key:
            self.on_status(self.state, f"loading speech model '{key}' (first time downloads it)…")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(key, device="cpu", compute_type="int8")
            self._model_key = key
        return self._model

    def _transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < SR * 0.3:
            return ""
        segs, _ = self._load().transcribe(
            audio, language=self.language, vad_filter=True, beam_size=1,
            condition_on_previous_text=False, initial_prompt=self._text[-200:] or None,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def _type(self, text: str) -> None:
        if not text:
            return
        from pynput import keyboard

        if self._typed:
            text = " " + text
        keyboard.Controller().type(text)
        self._typed += len(text)
        self._text += text
        self.on_status(self.state, self._text)

    def _close_stream(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
            self._buf = []
        return audio

    def _cut_phrase(self) -> np.ndarray | None:
        """Return a finished phrase (speech followed by a pause) from the buffer, if any."""
        with self._lock:
            if not self._buf:
                return None
            audio = np.concatenate(self._buf)
        n = len(audio) // FRAME
        if n < 10:
            return None
        rms = np.sqrt((audio[: n * FRAME].reshape(n, FRAME) ** 2).mean(1))
        thresh = max(0.01, float(np.percentile(rms, 20)) * 3)
        voiced = np.where(rms > thresh)[0]
        if len(voiced) and voiced[-1] >= n - 3:
            self._last_voice = time.monotonic()
        if not len(voiced):
            if len(audio) > SR * 3:  # only silence: keep the buffer short
                with self._lock:
                    self._buf = [audio[-SR // 2:]]
            return None
        trailing = (n - 1 - voiced[-1]) * FRAME / SR
        if trailing < PAUSE_S and len(audio) < SR * MAX_PHRASE_S:
            return None
        with self._lock:  # take what we analysed, keep anything that arrived since
            rest = np.concatenate(self._buf)[len(audio):]
            self._buf = [rest] if len(rest) else []
        return audio

    def _run(self) -> None:
        while True:
            try:
                cmd = self._cmds.get(timeout=0.1)
            except queue.Empty:
                cmd = None
            try:
                if cmd == "preload":
                    self._load()
                    self.on_status(self.state, "")
                elif cmd == "cancel" and self.state != "idle":
                    self._close_stream()
                    from pynput import keyboard

                    kb = keyboard.Controller()
                    for _ in range(self._typed):
                        kb.tap(keyboard.Key.backspace)
                    self.state = "idle"
                    self.on_status("cancelled", "")
                elif cmd in ("send", "stop") and self.state == "listening":
                    self.state = "finishing"
                    self._type(self._transcribe(self._close_stream()))
                    if cmd == "send":
                        from pynput import keyboard

                        time.sleep(0.05)
                        keyboard.Controller().tap(keyboard.Key.enter)
                    self.state = "idle"
                    self.on_status("sent" if cmd == "send" else "done", self._text)
                elif self.state == "listening":
                    phrase = self._cut_phrase()
                    if phrase is not None:
                        self._type(self._transcribe(phrase))
            except Exception as e:
                self._close_stream()
                self.state = "idle"
                self.on_status("error", str(e)[:150])
