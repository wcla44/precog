"""Camera thread: face/iris landmarks -> gaze feature vector, hand landmarks -> pinch gesture."""
from __future__ import annotations

import collections
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .gestures import SwipeDetector, classify

MODELS = Path(__file__).resolve().parents[2] / "models"  # repo root

# MediaPipe face mesh indices (subject's right eye, then left eye).
EYES = [
    dict(c1=33, c2=133, top=159, bot=145, iris=468),
    dict(c1=362, c2=263, top=386, bot=374, iris=473),
]
BLENDSHAPES = [
    f"eyeLook{d}{s}" for s in ("Left", "Right") for d in ("In", "Out", "Up", "Down")
]
FEATURE_NAMES = (
    ["rx", "ry", "ropen", "lx", "ly", "lopen"]
    + BLENDSHAPES
    + ["pitch", "yaw", "roll", "tx", "ty", "tz"]
)


@dataclass
class Frame:
    t: float
    features: np.ndarray | None  # None when no face
    pose: str | None  # hand pose: open / fist / pinch / other, None when no hand

    @property
    def hand(self) -> bool:
        return self.pose is not None

    @property
    def pinch(self) -> bool:
        return self.pose == "pinch"


def _eye_features(P: np.ndarray, e: dict) -> list[float]:
    a, b = P[e["c1"]], P[e["c2"]]
    axis = b - a
    n = float(np.linalg.norm(axis)) or 1.0
    ux = axis / n
    uy = np.array([-ux[1], ux[0]])
    d = P[e["iris"]] - (a + b) / 2
    openness = float(np.linalg.norm(P[e["top"]] - P[e["bot"]])) / n
    return [float(d @ ux) / n, float(d @ uy) / n, openness]


def _head_pose(m: np.ndarray) -> list[float]:
    R = m[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    tx, ty, tz = m[:3, 3]
    return [pitch, yaw, roll, tx / 10, ty / 10, tz / 10]


def extract_features(res, w: int, h: int) -> np.ndarray | None:
    if not res.face_landmarks:
        return None
    P = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    feats = _eye_features(P, EYES[0]) + _eye_features(P, EYES[1])
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
    feats += [bs.get(name, 0.0) for name in BLENDSHAPES]
    if res.facial_transformation_matrixes:
        feats += _head_pose(np.asarray(res.facial_transformation_matrixes[0]))
    else:
        feats += [0.0] * 6
    return np.asarray(feats, dtype=np.float64)


class Tracker:
    def __init__(self, camera: int = 0, hands: bool = True):
        self.camera = camera
        self.use_hands = hands
        self.history: collections.deque[Frame] = collections.deque(maxlen=300)
        self.swipes: queue.Queue[tuple[float, str]] = queue.Queue()
        self.mirrored = False
        self.latest: Frame | None = None
        self.fps = 0.0
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tracker")

    def start(self) -> "Tracker":
        # macOS: the camera must be opened (and permission requested) from the main thread
        self._cap = cv2.VideoCapture(self.camera)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self._cap.isOpened():
            self.error = ("cannot open camera: grant Camera access to your terminal app "
                          "(macOS: System Settings > Privacy & Security > Camera), then restart it")
            return self
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def frames_between(self, t0: float, t1: float) -> list[Frame]:
        with self._lock:
            return [f for f in self.history if t0 <= f.t <= t1 and f.features is not None]

    def _loop(self) -> None:
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        hand = None
        cap = self._cap
        last_ts, n, t_fps = 0, 0, time.monotonic()
        pose = None
        swipe = SwipeDetector()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.monotonic()
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            h, w = rgb.shape[:2]
            feats = extract_features(face.detect_for_video(img, ts), w, h)
            if self.use_hands and hand is None:
                hand = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=1,
                    )
                )
            if not self.use_hands:
                pose = None
            elif n % 2 == 0:  # hands every other frame: saves CPU
                hr = hand.detect_for_video(img, ts)
                lm = None
                if hr.hand_landmarks:
                    lm = np.array([[p.x * w / h, p.y] for p in hr.hand_landmarks[0]])  # square units
                pose = classify(lm) if lm is not None else None
                swipe.sign = -1 if self.mirrored else 1
                direction = swipe.update(t, lm[0] if lm is not None else None)
                if direction:
                    self.swipes.put((t, direction))
            f = Frame(t, feats, pose)
            with self._lock:
                self.history.append(f)
                self.latest = f
            n += 1
            if t - t_fps >= 1.0:
                self.fps = n / (t - t_fps)
                n, t_fps = 0, t
        cap.release()
