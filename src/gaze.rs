//! Gaze regression: eye/head features -> global screen point. Ridge regression with quadratic iris/head terms,
//! alpha picked by grouped k-fold (held-out calibration dots never seen in training).

use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const NF: usize = 14;
/// Columns that get squared / crossed: iris x,y of both eyes, pitch, yaw.
const QUAD: [usize; 6] = [0, 1, 3, 4, 6, 7];
const ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
const MAX_CLICK_SAMPLES: usize = 1500;
pub const CLICK_GROUP: i64 = 1000;

#[derive(Serialize, Deserialize, Default)]
pub struct GazeModel {
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

/// One-euro filter: smooth when still, responsive when moving.
pub struct OneEuro {
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
    state: Option<([f64; 2], [f64; 2], f64)>,
}

impl OneEuro {
    pub fn new() -> Self {
        Self { min_cutoff: 0.6, beta: 0.004, d_cutoff: 1.0, state: None }
    }

    fn alpha(cutoff: f64, dt: f64) -> f64 {
        let r = 2.0 * std::f64::consts::PI * cutoff * dt;
        r / (r + 1.0)
    }

    pub fn filter(&mut self, x: [f64; 2], t: f64) -> [f64; 2] {
        let Some((px, pdx, pt)) = self.state else {
            self.state = Some((x, [0.0; 2], t));
            return x;
        };
        let dt = (t - pt).max(1e-3);
        let ad = Self::alpha(self.d_cutoff, dt);
        let dx = [pdx[0] + ad * ((x[0] - px[0]) / dt - pdx[0]), pdx[1] + ad * ((x[1] - px[1]) / dt - pdx[1])];
        let cutoff = self.min_cutoff + self.beta * (dx[0].powi(2) + dx[1].powi(2)).sqrt();
        let a = Self::alpha(cutoff, dt);
        let nx = [px[0] + a * (x[0] - px[0]), px[1] + a * (x[1] - px[1])];
        self.state = Some((nx, dx, t));
        nx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
