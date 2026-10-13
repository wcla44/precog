//! Gaze regression: eye/head features -> global screen point. Ridge regression with quadratic iris/head terms,
//! alpha picked by grouped k-fold (held-out calibration dots never seen in training).

use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

pub const NF: usize = 14;
/// Bumped whenever the feature vector changes meaning: older saved models then trigger a recalibration.
pub const FEATURE_VERSION: u32 = 2;
/// Columns that get squared / crossed: iris x,y of both eyes, pitch, yaw.
const QUAD: [usize; 6] = [0, 1, 3, 4, 6, 7];
const ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
const MAX_CLICK_SAMPLES: usize = 1500;
pub const CLICK_GROUP: i64 = 1000;

#[derive(Serialize, Deserialize, Default)]
pub struct GazeModel {
    #[serde(default)]
    pub version: u32,
    pub f: Vec<[f64; NF]>,
    pub y: Vec<[f64; 2]>,
    pub g: Vec<i64>,
    pub w: Vec<f64>,
    /// Monitors (left, top, w, h) it was calibrated on.
    pub layout: Vec<(i32, i32, u32, u32)>,
    #[serde(skip)]
    fit: Option<Fit>,
}

struct Fit {
    mu: [f64; NF],
    sd: [f64; NF],
    coef: DMatrix<f64>, // (n_terms, 2)
    cv_error: f64,
}

impl GazeModel {
    pub fn cv_error(&self) -> f64 {
        self.fit.as_ref().map_or(f64::NAN, |f| f.cv_error)
    }

    pub fn len(&self) -> usize {
        self.f.len()
    }

    pub fn add(&mut self, feats: &[[f64; NF]], point: [f64; 2], group: i64, weight: f64) {
        for x in feats {
            self.f.push(*x);
            self.y.push(point);
            self.g.push(group);
            self.w.push(weight);
        }
        // cap implicit (click) samples, oldest first; calibration samples are kept
        let clicks = self.g.iter().filter(|&&g| g >= CLICK_GROUP).count();
        if clicks > MAX_CLICK_SAMPLES {
            let mut drop = clicks - MAX_CLICK_SAMPLES;
            let keep: Vec<bool> = self
                .g
                .iter()
                .map(|&g| {
                    if g >= CLICK_GROUP && drop > 0 {
                        drop -= 1;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            let mut i = 0;
            self.f.retain(|_| (keep[i], i += 1).0);
            i = 0;
            self.y.retain(|_| (keep[i], i += 1).0);
            i = 0;
            self.g.retain(|_| (keep[i], i += 1).0);
            i = 0;
            self.w.retain(|_| (keep[i], i += 1).0);
        }
    }

    fn expand(mu: &[f64; NF], sd: &[f64; NF], x: &[f64; NF]) -> Vec<f64> {
        let z: Vec<f64> = (0..NF).map(|i| (x[i] - mu[i]) / sd[i]).collect();
        let mut row = Vec::with_capacity(1 + NF + 21);
        row.push(1.0);
        row.extend_from_slice(&z);
        for i in 0..QUAD.len() {
            for j in i..QUAD.len() {
                row.push(z[QUAD[i]] * z[QUAD[j]]);
            }
        }
        row
    }

    fn solve(x: &DMatrix<f64>, y: &DMatrix<f64>, w: &[f64], rows: &[usize], alpha: f64) -> Option<DMatrix<f64>> {
        let p = x.ncols();
        let mut xtx = DMatrix::<f64>::zeros(p, p);
        let mut xty = DMatrix::<f64>::zeros(p, 2);
        for &r in rows {
            let xr = x.row(r);
            xtx += w[r] * xr.transpose() * xr;
            xty += w[r] * xr.transpose() * y.row(r);
        }
        for i in 1..p {
            xtx[(i, i)] += alpha; // don't penalise the bias
        }
        xtx.cholesky().map(|c| c.solve(&xty))
    }

    pub fn fit(&mut self) -> Result<f64> {
        let n = self.f.len();
        if n < 30 {
            bail!("not enough samples ({n})");
        }
        let mut mu = [0.0; NF];
        let mut sd = [0.0; NF];
        for i in 0..NF {
            mu[i] = self.f.iter().map(|x| x[i]).sum::<f64>() / n as f64;
            sd[i] = (self.f.iter().map(|x| (x[i] - mu[i]).powi(2)).sum::<f64>() / n as f64).sqrt() + 1e-6;
        }
        let rows: Vec<Vec<f64>> = self.f.iter().map(|x| Self::expand(&mu, &sd, x)).collect();
        let p = rows[0].len();
        let x = DMatrix::from_fn(n, p, |r, c| rows[r][c]);
        let y = DMatrix::from_fn(n, 2, |r, c| self.y[r][c]);

        let mut groups: Vec<i64> = self.g.clone();
        groups.sort_unstable();
        groups.dedup();
        groups.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0));
        let k = groups.len().min(5);
        let folds: Vec<Vec<i64>> = (0..k).map(|i| groups.iter().skip(i).step_by(k).copied().collect()).collect();

        let mut best = (f64::INFINITY, 1.0);
        for alpha in ALPHAS {
            let mut errs = Vec::with_capacity(n);
            for fold in &folds {
                let (te, tr): (Vec<usize>, Vec<usize>) = (0..n).partition(|&r| fold.contains(&self.g[r]));
                let Some(c) = Self::solve(&x, &y, &self.w, &tr, alpha) else { continue };
                for r in te {
                    let pred = x.row(r) * &c;
                    errs.push(((pred[0] - y[(r, 0)]).powi(2) + (pred[1] - y[(r, 1)]).powi(2)).sqrt());
                }
            }
            if errs.is_empty() {
                continue;
            }
            errs.sort_by(f64::total_cmp);
            let med = errs[errs.len() / 2];
            if med < best.0 {
                best = (med, alpha);
            }
        }
        let all: Vec<usize> = (0..n).collect();
        let Some(coef) = Self::solve(&x, &y, &self.w, &all, best.1) else { bail!("singular system") };
        self.fit = Some(Fit { mu, sd, coef, cv_error: best.0 });
        Ok(best.0)
    }

    pub fn predict(&self, x: &[f64; NF]) -> Option<(f64, f64)> {
        let f = self.fit.as_ref()?;
        let row = DVector::from_vec(Self::expand(&f.mu, &f.sd, x));
        let p = row.transpose() * &f.coef;
        Some((p[0], p[1]))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut m: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        m.fit()?;
        Ok(m)
    }
}

/// Fixation-aware gaze filter. Webcam gaze is noisy (~100-200 px per sample), and eyes move in two ways:
/// fixations (still, where averaging is right) and saccades (fast jumps, where averaging = lag).
/// Samples close to the current fixation are averaged; a few consecutive samples that agree on a new spot
/// end the fixation immediately (2 frames for big jumps, 3 otherwise), so the output never trails behind the eye.
pub struct FixationFilter {
    center: Option<[f64; 2]>,
    inliers: VecDeque<[f64; 2]>,
    pending: Vec<[f64; 2]>,
    outliers_in_a_row: u32,
    jitter: f64,
    shown: Option<[f64; 2]>,
    last_t: f64,
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn mean(pts: impl IntoIterator<Item = [f64; 2]>) -> [f64; 2] {
    let (mut s, mut n) = ([0.0f64; 2], 0.0f64);
    for p in pts {
        s[0] += p[0];
        s[1] += p[1];
        n += 1.0;
    }
    [s[0] / n.max(1.0), s[1] / n.max(1.0)]
}

impl FixationFilter {
    const MAX_INLIERS: usize = 12; // ~0.4 s of averaging at 30 fps
    const EASE: f64 = 0.035; // s: display easing, just enough to avoid a teleport flicker

    pub fn new(initial_jitter: f64) -> Self {
        Self {
            center: None,
            inliers: VecDeque::new(),
            pending: Vec::new(),
            outliers_in_a_row: 0,
            jitter: initial_jitter.clamp(30.0, 250.0),
            shown: None,
            last_t: 0.0,
        }
    }

    /// Estimated per-sample noise in px (mean distance of fixation samples to their centre).
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    fn radius(&self) -> f64 {
        (2.5 * self.jitter).clamp(60.0, 450.0)
    }

    fn start(&mut self, pts: &[[f64; 2]]) {
        self.inliers = pts.iter().copied().collect();
        self.center = Some(mean(pts.iter().copied()));
        self.pending.clear();
        self.outliers_in_a_row = 0;
    }

    pub fn reset(&mut self) {
        self.center = None;
        self.shown = None;
        self.inliers.clear();
        self.pending.clear();
    }

    pub fn update(&mut self, p: [f64; 2], t: f64) -> [f64; 2] {
        let r = self.radius();
        match self.center {
            None => self.start(&[p]),
            Some(c) if dist(p, c) <= r => {
                self.jitter = 0.98 * self.jitter + 0.02 * dist(p, c).max(5.0);
                self.inliers.push_back(p);
                while self.inliers.len() > Self::MAX_INLIERS {
                    self.inliers.pop_front();
                }
                self.center = Some(mean(self.inliers.iter().copied()));
                self.pending.clear();
                self.outliers_in_a_row = 0;
            }
            Some(c) => {
                self.pending.push(p);
                self.outliers_in_a_row += 1;
                let m = mean(self.pending.iter().copied());
                let agree = self.pending.iter().all(|&q| dist(q, m) <= r);
                let far = dist(m, c) > 2.0 * r;
                if agree && (self.pending.len() >= 3 || (far && self.pending.len() >= 2)) {
                    let pts = std::mem::take(&mut self.pending);
                    self.start(&pts); // saccade: jump now
                } else if !agree && self.pending.len() >= 3 {
                    self.pending.remove(0);
                }
                if self.outliers_in_a_row >= 6 {
                    // noisier than we thought: widen the fixation window and re-anchor
                    self.jitter = (self.jitter * 1.25).min(250.0);
                    let pts: Vec<_> = self.pending.iter().rev().take(3).copied().collect();
                    self.start(&pts);
                }
            }
        }
        let c = self.center.unwrap();
        let shown = self.shown.get_or_insert(c);
        let a = 1.0 - (-(t - self.last_t).max(0.0) / Self::EASE).exp();
        shown[0] += a * (c[0] - shown[0]);
        shown[1] += a * (c[1] - shown[1]);
        self.last_t = t;
        *shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noisy fixations with a saccade: steady during fixations, follows the jump within ~3 frames.
    #[test]
    fn fixation_filter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        use rand::Rng;
        let mut f = FixationFilter::new(100.0);
        let dt = 1.0 / 30.0;
        let (mut moves, mut last) = (Vec::new(), None::<[f64; 2]>);
        let mut lag = None;
        for i in 0..240 {
            let t = i as f64 * dt;
            let target = if i < 120 { [400.0, 300.0] } else { [1200.0, 700.0] };
            let p = [target[0] + rng.random_range(-110.0..110.0), target[1] + rng.random_range(-110.0..110.0)];
            let o = f.update(p, t);
            if (30..120).contains(&i) {
                if let Some(l) = last {
                    moves.push(dist(o, l));
                }
            }
            if i >= 120 && lag.is_none() && dist(o, target) < 100.0 {
                lag = Some((i - 120) as f64 * dt);
            }
            last = Some(o);
        }
        let m = moves.iter().sum::<f64>() / moves.len() as f64;
        println!("fixation motion {m:.1} px/frame, saccade lag {:?} s", lag);
        assert!(m < 12.0, "too jumpy during fixation: {m}");
        assert!(lag.is_some_and(|l| l <= 0.2), "saccade lag {lag:?}");
    }

    #[test]
    fn fits_synthetic() {
        let mut m = GazeModel::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        use rand::Rng;
        for g in 0..16 {
            let p = [rng.random_range(0.0..1500.0), rng.random_range(0.0..900.0)];
            let feats: Vec<[f64; NF]> = (0..30)
                .map(|_| {
                    let mut f = [0.0; NF];
                    for v in f.iter_mut() {
                        *v = rng.random_range(-1.0..1.0);
                    }
                    f[0] = p[0] / 1000.0 + rng.random_range(-0.02..0.02);
                    f[1] = p[1] / 1000.0 + rng.random_range(-0.02..0.02);
                    f
                })
                .collect();
            m.add(&feats, p, g, 1.0);
        }
        let err = m.fit().unwrap();
        assert!(err < 60.0, "cv error {err}");
    }
}
