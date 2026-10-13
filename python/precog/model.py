"""Gaze regression: features -> screen (x, y). Ridge regression with a few quadratic terms."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

MODEL_PATH = DATA_DIR / "gaze_model.npz"

# columns that get squared / crossed: iris x,y of both eyes, pitch, yaw
QUAD = [0, 1, 3, 4, 14, 15]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_CLICK_SAMPLES = 1500


class GazeModel:
    def __init__(self):
        self.F = np.zeros((0, 20))  # raw features
        self.Y = np.zeros((0, 2))  # screen points
        self.G = np.zeros(0, dtype=int)  # group id (calibration dot / click), for honest CV
        self.W = np.zeros(0)  # sample weight
        self.coef: np.ndarray | None = None
        self.mu = self.sd = None
        self.alpha = 1.0
        self.cv_error = float("nan")
        self.layout: list[tuple[int, int, int, int]] = []  # monitors (left, top, w, h) it was calibrated on

    @property
    def ready(self) -> bool:
        return self.coef is not None

    def add(self, feats: np.ndarray, point: tuple[float, float], group: int, weight: float = 1.0) -> None:
        self.F = np.vstack([self.F, feats])
        self.Y = np.vstack([self.Y, np.tile(point, (len(feats), 1))])
        self.G = np.concatenate([self.G, np.full(len(feats), group)])
        self.W = np.concatenate([self.W, np.full(len(feats), weight)])
        # cap implicit (click) samples, keep calibration ones (group < 1000)
        clicks = np.where(self.G >= 1000)[0]
        if len(clicks) > MAX_CLICK_SAMPLES:
            keep = np.ones(len(self.G), bool)
            keep[clicks[: len(clicks) - MAX_CLICK_SAMPLES]] = False
            self.F, self.Y, self.G, self.W = self.F[keep], self.Y[keep], self.G[keep], self.W[keep]

    def _expand(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.mu) / self.sd
        q = Z[:, QUAD]
        cross = [q[:, i] * q[:, j] for i in range(len(QUAD)) for j in range(i, len(QUAD))]
        return np.column_stack([np.ones(len(Z)), Z, *cross])

    @staticmethod
    def _solve(X, Y, W, alpha):
        Xw = X * W[:, None]
        reg = alpha * np.eye(X.shape[1])
        reg[0, 0] = 0  # don't penalise bias
        return np.linalg.solve(X.T @ Xw + reg, Xw.T @ Y)

    def fit(self) -> float:
        if len(self.F) < 30:
            raise ValueError("not enough samples")
        self.mu = self.F.mean(0)
        self.sd = self.F.std(0) + 1e-6
        X = self._expand(self.F)
        groups = np.unique(self.G)
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(groups), min(5, len(groups)))
        best = (math.inf, 1.0)
        for alpha in ALPHAS:  # grouped k-fold: held-out dots are never seen in training
            errs = []
            for fold in folds:
                te = np.isin(self.G, fold)
                c = self._solve(X[~te], self.Y[~te], self.W[~te], alpha)
                errs.append(np.linalg.norm(X[te] @ c - self.Y[te], axis=1))
            err = float(np.median(np.concatenate(errs)))
            if err < best[0]:
                best = (err, alpha)
        self.cv_error, self.alpha = best
        self.coef = self._solve(X, self.Y, self.W, self.alpha)
        return self.cv_error

    def predict(self, feats: np.ndarray) -> tuple[float, float]:
        x, y = (self._expand(feats[None, :]) @ self.coef)[0]
        return float(x), float(y)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, F=self.F, Y=self.Y, G=self.G, W=self.W, layout=np.array(self.layout).reshape(-1, 4))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "GazeModel":
        m = cls()
        d = np.load(path)
        m.F, m.Y, m.G, m.W = d["F"], d["Y"], d["G"], d["W"]
        m.layout = [tuple(int(v) for v in row) for row in d["layout"]] if "layout" in d else []
        m.fit()
        return m


class OneEuro:
    """One-euro filter: smooth when still, responsive when moving."""

    def __init__(self, min_cutoff=0.6, beta=0.004, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        t = time.monotonic() if t is None else t
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x = self.x + self._a(cutoff, dt) * (x - self.x)
        return self.x
