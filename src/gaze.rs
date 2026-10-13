//! Gaze regression: eye/head features -> global screen point. Ridge regression with quadratic iris/head terms,
//! alpha picked by grouped k-fold (held-out calibration dots never seen in training).
//!
//! Continuous recalibration: every mouse click is a labelled sample (people look at what they click). Clicks are
//! added with a recency weight and the model is refitted on each click; on top of that, `ClickCorrector` applies
//! an instant local correction from the residuals of the last clicks, which absorbs drift (posture, lighting).

use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

pub const NF: usize = 14;
/// Bumped whenever the feature vector changes meaning: older saved models then trigger a recalibration.
pub const FEATURE_VERSION: u32 = 4;
/// Columns that get squared / crossed: iris x,y of both eyes, head pitch/yaw.
const QUAD: [usize; 6] = [0, 1, 3, 4, 6, 7];
const ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
const MAX_CLICK_SAMPLES: usize = 3000;
/// Click samples fade with age (posture drifts): weight = FLOOR + (1 - FLOOR) * exp(-age / HALF_LIFE).
const CLICK_RECENCY_S: f64 = 1800.0;
const CLICK_RECENCY_FLOOR: f64 = 0.25;
pub const CLICK_GROUP: i64 = 1000;

#[derive(Serialize, Deserialize, Default)]
pub struct GazeModel {
    #[serde(default)]
    pub version: u32,
    pub f: Vec<[f64; NF]>,
    pub y: Vec<[f64; 2]>,
    pub g: Vec<i64>,
    pub w: Vec<f64>,
    /// Unix time each sample was recorded (for recency weighting of clicks).
    #[serde(default)]
    pub ts: Vec<f64>,
    /// Ridge strength chosen by cross-validation at the last full fit (refits on clicks reuse it).
    #[serde(default)]
    pub alpha: f64,
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
    /// Per-axis (offset, gain) undoing regression shrinkage: with noisy features, ridge predictions are pulled
    /// towards the middle of the screen (the gaze "can't reach" the edges). Fitted on held-out predictions.
    stretch: [(f64, f64); 2],
}

/// Least-squares target = offset + gain * prediction on one axis, gain limited to [1, 1.6]: only ever expands.
fn stretch_axis(pairs: &[([f64; 2], [f64; 2])], axis: usize) -> (f64, f64) {
    let n = pairs.len() as f64;
    if n < 10.0 {
        return (0.0, 1.0);
    }
    let mp = pairs.iter().map(|p| p.0[axis]).sum::<f64>() / n;
    let mt = pairs.iter().map(|p| p.1[axis]).sum::<f64>() / n;
    let cov = pairs.iter().map(|p| (p.0[axis] - mp) * (p.1[axis] - mt)).sum::<f64>();
    let var = pairs.iter().map(|p| (p.0[axis] - mp).powi(2)).sum::<f64>().max(1e-9);
    let gain = (cov / var).clamp(1.0, 1.6);
    (mt - gain * mp, gain)
}

pub fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
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
            self.ts.push(unix_now());
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
            i = 0;
            self.ts.retain(|_| (keep[i], i += 1).0);
        }
    }

    fn expand(mu: &[f64; NF], sd: &[f64; NF], x: &[f64; NF]) -> Vec<f64> {
        let z: Vec<f64> = (0..NF).map(|i| (x[i] - mu[i]) / sd[i]).collect();
        let mut row = Vec::with_capacity(1 + NF + QUAD.len() * (QUAD.len() + 1) / 2);
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

    /// Effective sample weights: clicks fade with age, calibration samples keep their weight.
    fn weights(&self) -> Vec<f64> {
        let now = unix_now();
        self.w
            .iter()
            .zip(&self.g)
            .enumerate()
            .map(|(i, (&w, &g))| {
                if g >= CLICK_GROUP {
                    let age = (now - self.ts.get(i).copied().unwrap_or(now)).max(0.0);
                    w * (CLICK_RECENCY_FLOOR + (1.0 - CLICK_RECENCY_FLOOR) * (-age / CLICK_RECENCY_S).exp())
                } else {
                    w
                }
            })
            .collect()
    }

    /// Full fit: picks the ridge strength by grouped cross-validation. Returns the CV median error (px).
    pub fn fit(&mut self) -> Result<f64> {
        self.fit_inner(true)
    }

    /// Quick refit with the current ridge strength (after each click; a few ms).
    pub fn refit(&mut self) -> Result<()> {
        self.fit_inner(false).map(|_| ())
    }

    fn fit_inner(&mut self, cross_validate: bool) -> Result<f64> {
        let n = self.f.len();
        if n < 30 {
            bail!("not enough samples ({n})");
        }
        self.ts.resize(n, unix_now());
        let w = self.weights();
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

        let all: Vec<usize> = (0..n).collect();
        if !cross_validate && self.alpha > 0.0 {
            let Some(coef) = Self::solve(&x, &y, &w, &all, self.alpha) else { bail!("singular system") };
            let cv_error = self.cv_error();
            let stretch = self.fit.as_ref().map_or([(0.0, 1.0); 2], |f| f.stretch);
            self.fit = Some(Fit { mu, sd, coef, cv_error, stretch });
            return Ok(cv_error);
        }
        let mut groups: Vec<i64> = self.g.clone();
        groups.sort_unstable();
        groups.dedup();
        groups.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0));
        let k = groups.len().min(5);
        let folds: Vec<Vec<i64>> = (0..k).map(|i| groups.iter().skip(i).step_by(k).copied().collect()).collect();

        let mut best = (f64::INFINITY, 1.0, Vec::new());
        for alpha in ALPHAS {
            let mut errs = Vec::with_capacity(n);
            let mut held_out = Vec::with_capacity(n); // (prediction, target) for rows never seen in training
            for fold in &folds {
                let (te, tr): (Vec<usize>, Vec<usize>) = (0..n).partition(|&r| fold.contains(&self.g[r]));
                let Some(c) = Self::solve(&x, &y, &w, &tr, alpha) else { continue };
                for r in te {
                    let pred = x.row(r) * &c;
                    errs.push(((pred[0] - y[(r, 0)]).powi(2) + (pred[1] - y[(r, 1)]).powi(2)).sqrt());
                    held_out.push(([pred[0], pred[1]], [y[(r, 0)], y[(r, 1)]]));
                }
            }
            if errs.is_empty() {
                continue;
            }
            errs.sort_by(f64::total_cmp);
            let med = errs[errs.len() / 2];
            if med < best.0 {
                best = (med, alpha, held_out);
            }
        }
        let Some(coef) = Self::solve(&x, &y, &w, &all, best.1) else { bail!("singular system") };
        self.alpha = best.1;
        let stretch = [0, 1].map(|a| stretch_axis(&best.2, a));
        self.fit = Some(Fit { mu, sd, coef, cv_error: best.0, stretch });
        Ok(best.0)
    }

    pub fn predict(&self, x: &[f64; NF]) -> Option<(f64, f64)> {
        let f = self.fit.as_ref()?;
        let row = DVector::from_vec(Self::expand(&f.mu, &f.sd, x));
        let p = row.transpose() * &f.coef;
        let [(ox, gx), (oy, gy)] = f.stretch;
        Some((ox + gx * p[0], oy + gy * p[1]))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        // check the feature version first: older files have a different vector length and would not parse
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        let version = v.get("version").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        if version != FEATURE_VERSION {
            bail!("gaze features changed since that calibration (v{version} -> v{FEATURE_VERSION})");
        }
        let mut m: Self = serde_json::from_value(v)?;
        m.fit()?;
        Ok(m)
    }
}

/// Instant local correction from recent clicks: residual = click position - model prediction, blended by
/// proximity on screen and age. One click right next to the gaze corrects about half of its residual.
pub struct ClickCorrector {
    clicks: VecDeque<(f64, [f64; 2], [f64; NF], [f64; 2])>, // (time, click position, mean features, residual)
}

impl ClickCorrector {
    const KEEP: usize = 40;
    const SIGMA: f64 = 350.0; // px: how far a click's correction reaches
    const TAU: f64 = 900.0; // s: how long it lasts
    const PRIOR: f64 = 1.0; // shrinkage: weight of "no correction"

    pub fn new() -> Self {
        Self { clicks: VecDeque::new() }
    }

    pub fn add(&mut self, t: f64, pos: [f64; 2], feats: [f64; NF]) {
        self.clicks.push_back((t, pos, feats, [0.0; 2]));
        while self.clicks.len() > Self::KEEP {
            self.clicks.pop_front();
        }
    }

    /// Recompute residuals against the current model (call after every refit).
    pub fn update_residuals(&mut self, model: &GazeModel) {
        for c in self.clicks.iter_mut() {
            if let Some((px, py)) = model.predict(&c.2) {
                c.3 = [c.1[0] - px, c.1[1] - py];
            }
        }
    }

    pub fn correction(&self, p: [f64; 2], t: f64) -> [f64; 2] {
        let (mut sum, mut wsum) = ([0.0; 2], Self::PRIOR);
        for (ct, pos, _, r) in &self.clicks {
            let d2 = (p[0] - pos[0]).powi(2) + (p[1] - pos[1]).powi(2);
            let k = (-d2 / (2.0 * Self::SIGMA * Self::SIGMA)).exp() * (-(t - ct).max(0.0) / Self::TAU).exp();
            sum[0] += k * r[0];
            sum[1] += k * r[1];
            wsum += k;
        }
        [sum[0] / wsum, sum[1] / wsum]
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

    /// Calibrate on a synthetic eye->screen mapping, then the user's posture drifts (feature offset):
    /// clicks must bring the error back down quickly.
    #[test]
    fn clicks_fix_drift() {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let feat = |p: [f64; 2], drift: f64, rng: &mut rand::rngs::StdRng| {
            let mut f = [0.0; NF];
            for v in f.iter_mut() {
                *v = rng.random_range(-0.05..0.05);
            }
            f[0] = p[0] / 1000.0 + drift + rng.random_range(-0.02..0.02);
            f[1] = p[1] / 1000.0 + rng.random_range(-0.02..0.02);
            f
        };
        let mut m = GazeModel::default();
        for g in 0..9 {
            let p = [200.0 + 500.0 * (g % 3) as f64, 150.0 + 300.0 * (g / 3) as f64];
            let fs: Vec<_> = (0..20).map(|_| feat(p, 0.0, &mut rng)).collect();
            m.add(&fs, p, g, 1.0);
        }
        m.fit().unwrap();
        let mut corr = ClickCorrector::new();
        let drift = 0.15; // = 150 px shift
        let err = |m: &GazeModel, corr: &ClickCorrector, rng: &mut rand::rngs::StdRng| {
            let mut e = 0.0;
            for _ in 0..50 {
                let p = [rng.random_range(200.0..1200.0), rng.random_range(150.0..750.0)];
                let (x, y) = m.predict(&feat(p, drift, rng)).unwrap();
                let c = corr.correction([x, y], 0.0);
                e += ((x + c[0] - p[0]).powi(2) + (y + c[1] - p[1]).powi(2)).sqrt() / 50.0;
            }
            e
        };
        let before = err(&m, &corr, &mut rng);
        for k in 0..15 {
            let p = [rng.random_range(200.0..1200.0), rng.random_range(150.0..750.0)];
            let fs: Vec<_> = (0..8).map(|_| feat(p, drift, &mut rng)).collect();
            let mut mean = [0.0; NF];
            for f in &fs {
                for i in 0..NF {
                    mean[i] += f[i] / fs.len() as f64;
                }
            }
            m.add(&fs, p, CLICK_GROUP + k, 1.0);
            corr.add(0.0, p, mean);
            m.refit().unwrap();
            corr.update_residuals(&m);
        }
        let after = err(&m, &corr, &mut rng);
        println!("mean error after drift: {before:.0}px -> after 15 clicks: {after:.0}px");
        assert!(after < before * 0.5, "{before} -> {after}");
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
