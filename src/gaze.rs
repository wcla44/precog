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

/// Landmark features (`tracker::features`), then two 60-pixel eye patches (`vision::eye_patches`).
pub const LAND: usize = 20;
pub const NF: usize = LAND + 2 * crate::vision::EYE_PATCH;
/// Bumped whenever the feature vector changes meaning: older saved models then trigger a recalibration.
pub const FEATURE_VERSION: u32 = 7;
/// Columns that get squared / crossed: iris x,y of both eyes, head pitch/yaw.
const QUAD: [usize; 6] = [0, 1, 3, 4, 6, 7];
const ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
const MAX_CLICK_SAMPLES: usize = 3000;
/// Click samples fade with age (posture drifts): weight = FLOOR + (1 - FLOOR) * exp(-age / HALF_LIFE).
const CLICK_RECENCY_S: f64 = 1800.0;
const CLICK_RECENCY_FLOOR: f64 = 0.25;
pub const CLICK_GROUP: i64 = 1000;
/// Geometric model: the eye-in-head features it regresses from (iris offsets + eyelid openness of both eyes, head
/// pitch / yaw); the head position and rotation enter through the geometry.
const GEO_DIM: usize = 8;

#[derive(Serialize, Deserialize, Default)]
pub struct GazeModel {
    #[serde(default)]
    pub version: u32,
    /// Feature dimension and the columns that get quadratic terms.
    pub dim: usize,
    pub quad: Vec<usize>,
    pub f: Vec<Vec<f64>>,
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
    /// Physical arrangement of the screens: the regression runs in mm (targets stay in pixels, `y`). None = pixels.
    #[serde(default)]
    pub phys: Option<crate::layout::Physical>,
    /// Geometric model (landmark features only): the regression predicts the gaze direction in the head frame and
    /// the screen point is where that ray meets the screens (see `geo`). Takes precedence over `phys`.
    #[serde(default)]
    pub geo: Option<crate::geo::Geo>,
    #[serde(skip)]
    pub opts: FitOpts,
    /// Camera focal length (px at 1280 px width) the features were computed with (set by the calibration).
    #[serde(default)]
    pub focal: Option<f64>,
    #[serde(skip)]
    fit: Option<Fit>,
}

struct Fit {
    mu: Vec<f64>,
    sd: Vec<f64>,
    coef: DMatrix<f64>, // (n_terms, 2)
    cv_error: f64,
    /// Per-axis (offset, gain) undoing regression shrinkage: with noisy features, ridge predictions are pulled
    /// towards the middle of the screen (the gaze "can't reach" the edges). Fitted on held-out predictions.
    stretch: [(f64, f64); 2],
}

/// Fitting variants (compared offline with `--replay` on recorded calibrations).
#[derive(Clone, Copy, Debug)]
pub struct FitOpts {
    /// Geometric model: undo ridge shrinkage on the predicted angles (the gaze "not reaching" the edges).
    pub stretch_angles: bool,
    /// Geometric model: keep the screens' poses and the depth scale physically plausible (tilt within 25 deg,
    /// depth scale, i.e. focal length relative to the assumed one, 0.45-2.2, offsets within 250 mm).
    pub bounded: bool,
    /// Add the eye patch pixels to the regression (WebGazer's features): their appearance carries the vertical gaze.
    pub eye_patches: bool,
}

/// Defaults from replayed real calibrations (dots train, pursuit tests; validation dots never learned from):
/// bounded geometry cut the error from 582 to 483 px; the eye patches from 329 to 219 px (vertical reach 0.36 ->
/// 0.62, validation 373 -> 324 px); the angle stretch did not help.
impl Default for FitOpts {
    fn default() -> Self {
        Self { stretch_angles: false, bounded: true, eye_patches: true }
    }
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

/// Face position in the camera image (centre x, centre y, width; see `tracker::features`): where the user sits
/// relative to the camera. Moving on the chair or tilting a laptop lid (with its camera) changes it.
const HEAD: [usize; 3] = [9, 10, 11];

/// Mean face position of some landmark feature vectors.
pub fn head_of<F: AsRef<[f64]>>(feats: &[F]) -> [f64; 3] {
    let n = feats.len().max(1) as f64;
    HEAD.map(|i| feats.iter().map(|f| f.as_ref()[i]).sum::<f64>() / n)
}

/// How far the user sits from a reference posture: about 1 = moved enough that the gaze is probably off.
pub fn head_shift(reference: [f64; 3], x: &[f64]) -> f64 {
    let [cx, cy, w] = HEAD.map(|i| x[i]);
    ((cx - reference[0]).abs() / 0.06).max((cy - reference[1]).abs() / 0.06).max((w / reference[2].max(1e-3) - 1.0).abs() / 0.12)
}

pub fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl GazeModel {
    /// Landmark features (`tracker::features`): fast (30 fps), noisy, head-pose sensitive.
    pub fn landmarks() -> Self {
        Self { version: FEATURE_VERSION, dim: NF, quad: QUAD.to_vec(), ..Default::default() }
    }

    /// UniGaze (pitch, yaw): slower (~10 Hz) but much more accurate and head-robust.
    pub fn unigaze() -> Self {
        Self { version: FEATURE_VERSION, dim: 2, quad: vec![0, 1], ..Default::default() }
    }

    pub fn ready(&self) -> bool {
        self.fit.is_some()
    }

    /// Follow a change of screens (plugged, unplugged, moved in the system layout), geometric model only: known
    /// screens keep their pose, every sample moves to the same physical spot in the new layout (dropped if its screen
    /// is gone), the per-user regression is kept. Returns the new screens, whose pose needs a calibration.
    pub fn adapt_to(&mut self, mons: &[crate::monitors::Monitor]) -> Result<Vec<crate::monitors::Monitor>> {
        let Some(old) = self.geo.clone() else { bail!("not a geometric model") };
        let (geo, new_screens) = old.adapt(mons);
        let moved: Vec<Option<[f64; 2]>> = self.y.iter().map(|t| geo.locate(old.point(t[0], t[1])).map(|(x, y)| [x, y])).collect();
        let keep: Vec<bool> = moved.iter().map(|m| m.is_some()).collect();
        let mut i = 0;
        self.f.retain(|_| (keep[i], i += 1).0);
        (i, self.y) = (0, moved.into_iter().flatten().collect());
        self.g.retain(|_| (keep[i], i += 1).0);
        i = 0;
        self.w.retain(|_| (keep[i], i += 1).0);
        i = 0;
        self.ts.retain(|_| (keep.get(i).copied().unwrap_or(true), i += 1).0);
        self.geo = Some(geo);
        self.layout = mons.iter().map(|m| m.layout()).collect();
        if self.len() >= 30 {
            self.refit()?;
        }
        Ok(new_screens)
    }

    /// Posture of the explicit calibration samples (landmark models only).
    pub fn calibrated_head(&self) -> Option<[f64; 3]> {
        let feats: Vec<&Vec<f64>> = self.f.iter().zip(&self.g).filter(|(_, &g)| g < CLICK_GROUP).map(|(f, _)| f).collect();
        (self.dim == NF && !feats.is_empty()).then(|| head_of(&feats))
    }

    pub fn cv_error(&self) -> f64 {
        self.fit.as_ref().map_or(f64::NAN, |f| f.cv_error)
    }

    pub fn len(&self) -> usize {
        self.f.len()
    }

    pub fn add<F: AsRef<[f64]>>(&mut self, feats: &[F], point: [f64; 2], group: i64, weight: f64) {
        for x in feats {
            debug_assert_eq!(x.as_ref().len(), self.dim);
            self.f.push(x.as_ref().to_vec());
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

    /// Regression inputs of one sample: the eye-in-head features in the geometric model, all features otherwise.
    fn row(&self, mu: &[f64], sd: &[f64], x: &[f64]) -> Vec<f64> {
        let mut row = if self.geo.is_some() {
            Self::expand(&QUAD, &mu[..GEO_DIM], &sd[..GEO_DIM], &x[..GEO_DIM])
        } else {
            let d = self.dim.min(LAND); // landmark (and UniGaze) features; the patches only when asked
            Self::expand(&self.quad, &mu[..d], &sd[..d], &x[..d])
        };
        if self.opts.eye_patches && self.dim == NF {
            row.extend((LAND..NF).map(|i| (x[i] - mu[i]) / sd[i]));
        }
        row
    }

    /// Fit the screens' 3D poses and the depth scale of the geometric model to the samples. Each plausible
    /// arrangement (`layout::Physical::candidates`, the system's own first) is a starting point refined by
    /// Nelder-Mead on the cross-validated error; the system's arrangement is kept unless another one is clearly
    /// better (it is context, not truth). Returns the error (px) and a description.
    pub fn fit_geometry(&mut self, mons: &[crate::monitors::Monitor]) -> Result<(f64, String)> {
        use crate::geo::Geo;
        use crate::layout::Physical;
        let mons = &crate::monitors::camera_first(mons);
        let n = self.f.len();
        if n < 30 || self.dim != NF {
            bail!("not enough samples ({n})");
        }
        // ridge strength for the search: picked by cross-validation on the system arrangement (a fixed, too strong
        // one shrinks the predicted angles, and the search then compensates by distorting the geometry)
        self.geo = Some(Geo::new(&Physical::candidates(mons)[0], mons, 1.0, vec![[0.0; 4]; mons.len().saturating_sub(1)]));
        self.fit()?;
        let alpha = self.alpha;
        self.ts.resize(n, unix_now());
        let w = self.weights();
        let mut mu = vec![0.0; NF];
        let mut sd = vec![0.0; NF];
        for i in 0..NF {
            mu[i] = self.f.iter().map(|x| x[i]).sum::<f64>() / n as f64;
            sd[i] = (self.f.iter().map(|x| (x[i] - mu[i]).powi(2)).sum::<f64>() / n as f64).sqrt() + 1e-6;
        }
        let rows: Vec<Vec<f64>> = self.f.iter().map(|x| self.row(&mu, &sd, x)).collect();
        let x = DMatrix::from_fn(n, rows[0].len(), |r, c| rows[r][c]);
        let mut groups: Vec<i64> = self.g.clone();
        groups.sort_unstable();
        groups.dedup();
        groups.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0));
        let k = groups.len().min(4);
        let folds: Vec<(Vec<usize>, Vec<usize>)> = (0..k)
            .map(|i| {
                let fold: Vec<i64> = groups.iter().skip(i).step_by(k).copied().collect();
                (0..n).partition(|&r| !fold.contains(&self.g[r]))
            })
            .collect();
        let cv = |geo: &Geo| -> f64 {
            let y = DMatrix::from_fn(n, 2, |r, c| geo.angles_to(&self.f[r], self.y[r][0], self.y[r][1])[c]);
            let mut errs = Vec::with_capacity(n);
            for (tr, te) in &folds {
                let Some(c) = Self::solve(&x, &y, &w, tr, alpha) else { return f64::INFINITY };
                for &r in te {
                    let p = x.row(r) * &c;
                    let e = geo.hit(&self.f[r], [p[0], p[1]]).map_or(1e5, |(px, py)| (px - self.y[r][0]).hypot(py - self.y[r][1]));
                    errs.push(if e.is_finite() { e } else { 1e5 });
                }
            }
            // mean of the best 80%: smoother than the median for the simplex search, still robust to outliers
            errs.sort_by(f64::total_cmp);
            let keep = &errs[..(errs.len() * 4 / 5).max(1)];
            keep.iter().sum::<f64>() / keep.len() as f64
        };
        let n_sec = mons.len().saturating_sub(1);
        // parameters in natural units: ln(depth scale); per other screen right / down / towards the user (100 mm)
        // and tilt (0.3 rad)
        let bounded = self.opts.bounded;
        let squash = |v: f64, lim: f64| if bounded { lim * (v / lim).tanh() } else { v };
        let build = |phys: &Physical, p: &[f64]| {
            let params = (0..n_sec)
                .map(|i| {
                    let mm = |k: usize| squash(p[k + 4 * i] * 100.0, 250.0);
                    [mm(1), mm(2), mm(3), squash(p[4 + 4 * i] * 0.3, 25f64.to_radians())]
                })
                .collect();
            // the depth scale is the camera focal length relative to the assumed one (PnP depth scales with it):
            // webcams range from narrow to wide, so it gets a wide range (0.45-2.2)
            Geo::new(phys, mons, squash(p[0], 0.8).exp(), params)
        };
        let all: Vec<usize> = (0..n).collect();
        // Levenberg-Marquardt on the training residuals (px): for given screen poses the angle regression has a
        // closed form, so the residuals are a smooth function of the poses. Pseudo-Huber (150 px) against outlier
        // frames, and a weak prior pulling towards the starting arrangement.
        let residuals = |phys: &Physical, p: &[f64]| -> Vec<f64> {
            let geo = build(phys, p);
            let y = DMatrix::from_fn(n, 2, |r, c| geo.angles_to(&self.f[r], self.y[r][0], self.y[r][1])[c]);
            let mut out = Vec::with_capacity(2 * n + p.len());
            let Some(c) = Self::solve(&x, &y, &w, &all, alpha) else { return vec![1e5; 2 * n + p.len()] };
            for r in 0..n {
                let q = x.row(r) * &c;
                let (hx, hy) = geo.hit(&self.f[r], [q[0], q[1]]).unwrap_or((1e5, 1e5));
                for e in [hx - self.y[r][0], hy - self.y[r][1]] {
                    let e = if e.is_finite() { e } else { 1e5 };
                    out.push(w[r].sqrt() * e / (1.0 + (e / 150.0).powi(2)).powf(0.25));
                }
            }
            let scale = (n as f64).sqrt() * 5.0;
            out.extend(p.iter().map(|v| v * scale));
            out
        };
        let candidates = Physical::candidates(mons);
        let zero = vec![0.0; 1 + 4 * n_sec];
        let mut start: Vec<(f64, usize)> = candidates.iter().enumerate().map(|(i, c)| (cv(&build(c, &zero)), i)).collect();
        let system_start = start[0];
        start[1..].sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut results: Vec<(f64, Geo)> = Vec::new();
        for &(_, i) in std::iter::once(&system_start).chain(start[1..].iter().take(2)) {
            let phys = &candidates[i];
            let p = levenberg_marquardt(|p| residuals(phys, p), &zero, 30);
            let geo = build(phys, &p);
            results.push((cv(&geo), geo));
        }
        let system = results[0].clone();
        let best = results.iter().min_by(|a, b| a.0.total_cmp(&b.0)).cloned().unwrap_or_else(|| system.clone());
        let (err, geo) = if best.0 < 0.85 * system.0 { best } else { system };
        let describe = format!(
            "{} (camera focal ~{:.0} px at 1280 px width{})",
            if geo.label.is_empty() { "single screen" } else { &geo.label },
            1000.0 * geo.depth_scale,
            geo.params
                .iter()
                .map(|q| format!("; other screen moved {:+.0}/{:+.0}/{:+.0} mm, tilted {:+.0} deg", q[0], q[1], q[2], q[3].to_degrees()))
                .collect::<String>()
        );
        self.geo = Some(geo);
        self.phys = None;
        Ok((err, describe))
    }

    fn expand(quad: &[usize], mu: &[f64], sd: &[f64], x: &[f64]) -> Vec<f64> {
        let z: Vec<f64> = (0..mu.len()).map(|i| (x[i] - mu[i]) / sd[i]).collect();
        let mut row = Vec::with_capacity(1 + z.len() + quad.len() * (quad.len() + 1) / 2);
        row.push(1.0);
        row.extend_from_slice(&z);
        for i in 0..quad.len() {
            for j in i..quad.len() {
                row.push(z[quad[i]] * z[quad[j]]);
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
        let d = self.dim;
        let mut mu = vec![0.0; d];
        let mut sd = vec![0.0; d];
        for i in 0..d {
            mu[i] = self.f.iter().map(|x| x[i]).sum::<f64>() / n as f64;
            sd[i] = (self.f.iter().map(|x| (x[i] - mu[i]).powi(2)).sum::<f64>() / n as f64).sqrt() + 1e-6;
        }
        let rows: Vec<Vec<f64>> = self.f.iter().map(|x| self.row(&mu, &sd, x)).collect();
        let p = rows[0].len();
        let x = DMatrix::from_fn(n, p, |r, c| rows[r][c]);
        // regress gaze angles (geometric model), or physical mm, or pixels
        let target: Vec<[f64; 2]> = match (&self.geo, &self.phys) {
            (Some(g), _) => self.f.iter().zip(&self.y).map(|(f, t)| g.angles_to(f, t[0], t[1])).collect(),
            (None, Some(ph)) => self.y.iter().map(|t| ph.to_mm(t[0], t[1])).map(|(a, b)| [a, b]).collect(),
            (None, None) => self.y.clone(),
        };
        let y = DMatrix::from_fn(n, 2, |r, c| target[r][c]);
        let to_px = |r: usize, p: [f64; 2]| match (&self.geo, &self.phys) {
            (Some(g), _) => g.hit(&self.f[r], p).unwrap_or((f64::NAN, f64::NAN)),
            (None, Some(ph)) => ph.to_px(p[0], p[1]),
            (None, None) => (p[0], p[1]),
        };

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
                    let (px, py) = to_px(r, [pred[0], pred[1]]); // errors in pixels, whatever the regression space
                    let e = (px - self.y[r][0]).hypot(py - self.y[r][1]);
                    errs.push(if e.is_finite() { e } else { 1e5 });
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
        // the stretch undoes ridge shrinkage in screen space; angles go through the geometry instead
        let stretch = if self.geo.is_some() && !self.opts.stretch_angles { [(0.0, 1.0); 2] } else { [0, 1].map(|a| stretch_axis(&best.2, a)) };
        self.fit = Some(Fit { mu, sd, coef, cv_error: best.0, stretch });
        Ok(best.0)
    }

    pub fn predict(&self, x: &[f64]) -> Option<(f64, f64)> {
        let f = self.fit.as_ref()?;
        let row = DVector::from_vec(self.row(&f.mu, &f.sd, x));
        let p = row.transpose() * &f.coef;
        let [(ox, gx), (oy, gy)] = f.stretch;
        if let Some(g) = &self.geo {
            return g.hit(x, [ox + gx * p[0], oy + gy * p[1]]);
        }
        let (px, py) = (ox + gx * p[0], oy + gy * p[1]);
        Some(match &self.phys {
            Some(ph) => ph.to_px(px, py),
            None => (px, py),
        })
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

/// Least squares: minimise |r(x)|^2 from `x0` with Levenberg-Marquardt (numeric Jacobian).
pub fn levenberg_marquardt(r: impl Fn(&[f64]) -> Vec<f64>, x0: &[f64], iters: usize) -> Vec<f64> {
    let d = x0.len();
    let mut x = x0.to_vec();
    let mut e = r(&x);
    let mut cost: f64 = e.iter().map(|v| v * v).sum();
    let mut lambda = 1e-2;
    for _ in 0..iters {
        let m = e.len();
        let mut j = DMatrix::<f64>::zeros(m, d);
        for k in 0..d {
            let mut xp = x.clone();
            xp[k] += 1e-4;
            let ep = r(&xp);
            for i in 0..m {
                j[(i, k)] = (ep[i] - e[i]) / 1e-4;
            }
        }
        let ev = DVector::from_vec(e.clone());
        let jtj = j.transpose() * &j;
        let jte = j.transpose() * ev;
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj.clone();
            for k in 0..d {
                a[(k, k)] *= 1.0 + lambda;
                a[(k, k)] += 1e-9;
            }
            let Some(step) = a.cholesky().map(|c| c.solve(&jte)) else { break };
            let xn: Vec<f64> = (0..d).map(|k| x[k] - step[k]).collect();
            let en = r(&xn);
            let cn: f64 = en.iter().map(|v| v * v).sum();
            if cn < cost {
                let small = step.norm() < 1e-6;
                (x, e, cost) = (xn, en, cn);
                lambda = (lambda * 0.3).max(1e-7);
                improved = true;
                if small {
                    return x;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    x
}

/// Instant local correction from recent clicks: residual = click position - model prediction, blended by
/// proximity on screen and age. One click right next to the gaze corrects about half of its residual.
pub struct ClickCorrector<T> {
    clicks: VecDeque<(f64, [f64; 2], T, [f64; 2])>, // (time, click position, model inputs at the click, residual)
}

impl<T> ClickCorrector<T> {
    const KEEP: usize = 40;
    const SIGMA: f64 = 350.0; // px: how far a click's correction reaches
    const TAU: f64 = 900.0; // s: how long it lasts
    const PRIOR: f64 = 1.0; // shrinkage: weight of "no correction"

    pub fn new() -> Self {
        Self { clicks: VecDeque::new() }
    }

    pub fn add(&mut self, t: f64, pos: [f64; 2], feats: T) {
        self.clicks.push_back((t, pos, feats, [0.0; 2]));
        while self.clicks.len() > Self::KEEP {
            self.clicks.pop_front();
        }
    }

    /// Recompute residuals against the current predictor (call after every refit).
    pub fn update_residuals(&mut self, predict: impl Fn(&T) -> Option<(f64, f64)>) {
        for c in self.clicks.iter_mut() {
            if let Some((px, py)) = predict(&c.2) {
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

/// Adaptive Kalman filter for gaze. Eyes alternate between fixations (still: average the noise away) and
/// saccades (fast jumps: follow at once). A plain Kalman filter would lag on saccades or be noisy on fixations, so:
/// - fixation: random-walk Kalman (2D, isotropic); the measurement noise is estimated online from the innovations;
/// - saccade: when 2 consecutive measurements fall far outside the predicted distribution (> 4 sigma) and agree
///   with each other, the state is reset onto them (~2 frames of latency);
/// - a single outlier frame (blink, bad landmark, head jolt) is ignored.
pub struct GazeKalman {
    x: Option<[f64; 2]>,
    p: f64,         // state variance (px^2, per axis)
    r: f64,         // measurement variance (px^2, per axis), learned
    pending: Vec<[f64; 2]>,
    outliers: u32,
    shown: Option<[f64; 2]>,
    last_t: f64,
}

impl GazeKalman {
    const Q: f64 = 40.0 * 40.0; // px^2/s: how fast the fixation point may drift
    const GATE: f64 = 16.0; // squared Mahalanobis distance (2D) beyond which a sample is an outlier
    const EASE: f64 = 0.035; // s: display easing, just enough to avoid a teleport flicker

    pub fn new(initial_noise: f64) -> Self {
        let s = initial_noise.clamp(20.0, 250.0);
        Self { x: None, p: s * s, r: s * s, pending: Vec::new(), outliers: 0, shown: None, last_t: 0.0 }
    }

    /// Estimated per-sample noise in px (one axis).
    pub fn jitter(&self) -> f64 {
        self.r.sqrt()
    }

    pub fn reset(&mut self) {
        self.x = None;
        self.shown = None;
        self.pending.clear();
        self.outliers = 0;
    }

    fn jump(&mut self, to: [f64; 2]) {
        self.x = Some(to);
        self.p = self.r / 2.0;
        self.pending.clear();
        self.outliers = 0;
    }

    pub fn update(&mut self, z: [f64; 2], t: f64) -> [f64; 2] {
        let dt = (t - self.last_t).clamp(0.0, 0.5);
        match self.x {
            None => self.jump(z),
            Some(x) => {
                self.p += Self::Q * dt; // predict (random walk)
                let s = self.p + self.r;
                let nu = [z[0] - x[0], z[1] - x[1]];
                let d2 = (nu[0] * nu[0] + nu[1] * nu[1]) / s;
                if d2 <= Self::GATE {
                    let k = self.p / s;
                    self.x = Some([x[0] + k * nu[0], x[1] + k * nu[1]]);
                    self.p *= 1.0 - k;
                    // learn the measurement noise from fixation innovations (E[nu^2] per axis = S)
                    let r_obs = (nu[0] * nu[0] + nu[1] * nu[1]) / 2.0 - self.p;
                    self.r = (0.98 * self.r + 0.02 * r_obs.max(0.0)).clamp(15.0 * 15.0, 250.0 * 250.0);
                    self.pending.clear();
                    self.outliers = 0;
                } else {
                    self.pending.push(z);
                    self.outliers += 1;
                    let m = mean(self.pending.iter().copied());
                    let sigma = self.r.sqrt();
                    let agree = self.pending.iter().all(|&q| dist(q, m) <= 3.0 * sigma);
                    if agree && self.pending.len() >= 2 {
                        self.jump(m); // saccade
                    } else if !agree && self.pending.len() >= 2 {
                        self.pending.remove(0);
                    }
                    if self.outliers >= 6 {
                        // much noisier than estimated: widen and re-anchor
                        self.r = (self.r * 2.0).min(250.0 * 250.0);
                        let pts: Vec<_> = self.pending.iter().rev().take(3).copied().collect();
                        self.jump(mean(pts));
                    }
                }
            }
        }
        let c = self.x.unwrap();
        let shown = self.shown.get_or_insert(c);
        let a = 1.0 - (-(t - self.last_t).max(0.0) / Self::EASE).exp();
        shown[0] += a * (c[0] - shown[0]);
        shown[1] += a * (c[1] - shown[1]);
        self.last_t = t;
        *shown
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Noisy fixations, a single-frame spike, and a saccade: steady during fixations, ignores the spike,
    /// follows the saccade within ~2-3 frames.
    #[test]
    fn kalman_filter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        use rand::Rng;
        let mut f = GazeKalman::new(100.0);
        let dt = 1.0 / 30.0;
        let (mut moves, mut last, mut spike_shift) = (Vec::new(), None::<[f64; 2]>, 0.0);
        let mut lag = None;
        for i in 0..240 {
            let t = i as f64 * dt;
            let target = if i < 120 { [400.0, 300.0] } else { [1200.0, 700.0] };
            let mut p = [target[0] + rng.random_range(-110.0..110.0), target[1] + rng.random_range(-110.0..110.0)];
            if i == 80 {
                p = [1500.0, 100.0]; // one bad frame
            }
            let o = f.update(p, t);
            if i == 81 {
                spike_shift = dist(o, last.unwrap());
            }
            if (30..120).contains(&i) && i != 81 {
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
        println!("kalman: fixation motion {m:.1} px/frame, spike moved it {spike_shift:.0} px, saccade lag {lag:?} s");
        assert!(m < 10.0, "too jumpy during fixation: {m}");
        assert!(spike_shift < 30.0, "spike not rejected: {spike_shift}");
        assert!(lag.is_some_and(|l| l <= 0.15), "saccade lag {lag:?}");
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
        let mut m = GazeModel::landmarks();
        for g in 0..9 {
            let p = [200.0 + 500.0 * (g % 3) as f64, 150.0 + 300.0 * (g / 3) as f64];
            let fs: Vec<_> = (0..20).map(|_| feat(p, 0.0, &mut rng)).collect();
            m.add(&fs, p, g, 1.0);
        }
        m.fit().unwrap();
        let mut corr = ClickCorrector::new();
        let drift = 0.15; // = 150 px shift
        let err = |m: &GazeModel, corr: &ClickCorrector<[f64; NF]>, rng: &mut rand::rngs::StdRng| {
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
            corr.update_residuals(|f: &[f64; NF]| m.predict(f));
        }
        let after = err(&m, &corr, &mut rng);
        println!("mean error after drift: {before:.0}px -> after 15 clicks: {after:.0}px");
        assert!(after < before * 0.5, "{before} -> {after}");
    }

    /// Simulated user: true screens differ from the system layout (external monitor shifted, pushed back, tilted),
    /// the head moves and turns, iris features respond to the true eye-in-head angles. The geometric model must
    /// recover the geometry and stay accurate at new head positions, where a pixel regression fails.
    #[test]
    fn geometric_model_handles_head_movement() {
        use crate::geo::Geo;
        use crate::layout::Physical;
        use crate::monitors::Monitor;
        use nalgebra::{Rotation3, Vector3};
        use rand::Rng;
        let mons = vec![
            Monitor { left: 0, top: 1440, width: 1920, height: 1080, primary: true, mm: Some((294.0, 165.0)), builtin: true },
            Monitor { left: 0, top: 0, width: 2560, height: 1440, primary: false, mm: Some((597.0, 336.0)), builtin: false },
        ];
        let truth = Geo::new(&Physical::candidates(&mons)[0], &mons, 1.0, vec![[60.0, -20.0, -150.0, 0.2]]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        // one sample: features for a user at `eye`, head rotated `rv`, looking at (x, y)
        let sample = |rng: &mut rand::rngs::StdRng, x: f64, y: f64, eye: [f64; 3], rv: [f64; 3]| {
            let mut f = vec![0.0; NF];
            f[14..17].copy_from_slice(&rv);
            f[17..20].copy_from_slice(&eye);
            let [yaw, pitch] = truth.angles_to(&f, x, y);
            let r = Rotation3::new(Vector3::new(rv[0], rv[1], rv[2]));
            let (hp, hy, _) = r.euler_angles();
            let noise = |rng: &mut rand::rngs::StdRng| rng.random_range(-0.004..0.004);
            // iris offsets ~ sin of the eye rotation (eye width units), lids open with upward gaze
            f[0] = 0.35 * yaw.sin() + noise(rng);
            f[1] = 0.30 * pitch.sin() + noise(rng);
            f[2] = 0.30 - 0.1 * pitch;
            f[3] = 0.35 * yaw.sin() + noise(rng);
            f[4] = 0.30 * pitch.sin() + noise(rng);
            f[5] = 0.30 - 0.1 * pitch;
            f[6] = hp;
            f[7] = hy;
            f
        };
        let head = |rng: &mut rand::rngs::StdRng, spread: f64| {
            let eye = [rng.random_range(-spread..spread), rng.random_range(40.0..120.0), rng.random_range(500.0..700.0)];
            let rv = [rng.random_range(-0.15..0.15), rng.random_range(-0.15..0.15), rng.random_range(-0.05..0.05)];
            (eye, rv)
        };
        let mut m = GazeModel::landmarks();
        for (g, (mi, fx, fy)) in [0usize, 1].iter().flat_map(|&mi| (0..9).map(move |k| (mi, 0.05 + 0.45 * (k % 3) as f64, 0.05 + 0.45 * (k / 3) as f64))).enumerate() {
            let mo = mons[mi];
            let (x, y) = (mo.left as f64 + fx * mo.width as f64, mo.top as f64 + fy * mo.height as f64);
            let (eye, rv) = head(&mut rng, 80.0);
            let feats: Vec<Vec<f64>> = (0..12).map(|_| sample(&mut rng, x, y, eye, rv)).collect();
            m.add(&feats, [x, y], g as i64, 1.0);
        }
        let mut t = GazeModel::landmarks();
        (t.f, t.y, t.g, t.w) = (m.f.clone(), m.y.clone(), m.g.clone(), m.w.clone());
        t.geo = Some(truth.clone());
        let truth_cv = t.fit().unwrap();
        let (err, what) = m.fit_geometry(&mons).unwrap();
        let found_cv = m.fit().unwrap();
        println!("cv with the true geometry {truth_cv:.0}px, with the fitted one {found_cv:.0}px");
        // new head positions, never seen
        let mut errs = Vec::new();
        for _ in 0..200 {
            let mi = rng.random_range(0..2);
            let mo = mons[mi];
            let (x, y) = (mo.left as f64 + rng.random_range(0.0..1.0) * mo.width as f64, mo.top as f64 + rng.random_range(0.0..1.0) * mo.height as f64);
            let (eye, rv) = head(&mut rng, 120.0);
            let (px, py) = m.predict(&sample(&mut rng, x, y, eye, rv)).unwrap();
            errs.push((px - x).hypot(py - y));
        }
        errs.sort_by(f64::total_cmp);
        let med = errs[errs.len() / 2];
        // baseline: the same samples, regressed straight to pixels
        let mut px = GazeModel::landmarks();
        px.f = m.f.clone();
        px.y = m.y.clone();
        px.g = m.g.clone();
        px.w = m.w.clone();
        px.fit().unwrap();
        let mut base = Vec::new();
        for _ in 0..200 {
            let mi = rng.random_range(0..2);
            let mo = mons[mi];
            let (x, y) = (mo.left as f64 + rng.random_range(0.0..1.0) * mo.width as f64, mo.top as f64 + rng.random_range(0.0..1.0) * mo.height as f64);
            let (eye, rv) = head(&mut rng, 120.0);
            let (qx, qy) = px.predict(&sample(&mut rng, x, y, eye, rv)).unwrap();
            base.push((qx - x).hypot(qy - y));
        }
        base.sort_by(f64::total_cmp);
        println!("geometry: {what}, cv {err:.0}px, new head positions: median {med:.0}px (pixel regression: {:.0}px)", base[base.len() / 2]);
        assert!(med < 80.0, "median error at new head positions {med:.0}px ({what})");
    }

    #[test]
    fn fits_synthetic() {
        let mut m = GazeModel::landmarks();
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
