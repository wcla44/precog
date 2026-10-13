//! Gaze cloud: a soft, slowly drifting cloud of particles around where the user looks, instead of a ring that
//! jumps with every noisy gaze sample. The center follows the *recent average* gaze through a critically damped
//! spring, and the cloud grows when the gaze is noisy, so it reads as "around here" rather than "exactly here".

use egui::{Color32, Painter, Pos2, Stroke};
use rand::Rng;
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Window side, in points: must hold the largest cloud plus its ripples.
pub const SIZE: f32 = 560.0;

const AVG_WINDOW: f64 = 0.6; // s of gaze averaged into the target
const SPREAD_WINDOW: f64 = 1.2; // s of gaze used to estimate the noise
const SPRING_HZ: f64 = 0.55; // natural frequency: lower = lazier
const MIN_R: f32 = 70.0;
const MAX_R: f32 = 200.0;
const COLOR: (u8, u8, u8) = (130, 190, 255);

struct Particle {
    angle: f32,  // initial angle
    radius: f32, // 0..1, fraction of the cloud radius
    speed: f32,  // angular speed, rad/s (signed)
    phase: f32,
    size: f32,
}

pub struct GazeCloud {
    samples: VecDeque<(f64, f64, f64)>,
    pos: Option<[f64; 2]>,
    vel: [f64; 2],
    radius: f32,
    particles: Vec<Particle>,
}

impl GazeCloud {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let particles = (0..160)
            .map(|_| {
                let r: f32 = rng.random::<f32>().powf(0.7); // denser towards the middle
                Particle {
                    angle: rng.random::<f32>() * TAU,
                    radius: r,
                    speed: (0.08 + rng.random::<f32>() * 0.3) * if rng.random::<bool>() { 1.0 } else { -1.0 },
                    phase: rng.random::<f32>() * TAU,
                    size: 1.0 + 1.8 * (1.0 - r) + rng.random::<f32>(),
                }
            })
            .collect();
        Self { samples: VecDeque::new(), pos: None, vel: [0.0; 2], radius: MIN_R, particles }
    }

    /// Feed the latest gaze point (or None when the face is lost). Returns the cloud center to draw at.
    pub fn update(&mut self, t: f64, dt: f64, gaze: Option<(f64, f64)>) -> Option<[f64; 2]> {
        if let Some((x, y)) = gaze {
            self.samples.push_back((t, x, y));
        }
        while self.samples.front().is_some_and(|s| t - s.0 > SPREAD_WINDOW) {
            self.samples.pop_front();
        }
        if self.samples.is_empty() {
            self.pos = None;
            return None;
        }
        let mean = |win: f64| {
            let pts: Vec<_> = self.samples.iter().filter(|s| t - s.0 <= win).collect();
            let n = pts.len().max(1) as f64;
            (pts.iter().map(|s| s.1).sum::<f64>() / n, pts.iter().map(|s| s.2).sum::<f64>() / n)
        };
        let (tx, ty) = mean(AVG_WINDOW);
        let (mx, my) = mean(SPREAD_WINDOW);
        let n = self.samples.len() as f64;
        let spread = (self.samples.iter().map(|s| (s.1 - mx).powi(2) + (s.2 - my).powi(2)).sum::<f64>() / n).sqrt();
        let target_r = (MIN_R + spread as f32 * 0.8).clamp(MIN_R, MAX_R);
        self.radius += (target_r - self.radius) * (1.0 - (-dt as f32 * 1.5).exp()); // ease the size too

        let pos = self.pos.get_or_insert([tx, ty]);
        // critically damped spring towards the averaged target: smooth, no overshoot
        let w = std::f64::consts::TAU * SPRING_HZ;
        let dt = dt.min(0.1);
        for (i, target) in [tx, ty].into_iter().enumerate() {
            let acc = w * w * (target - pos[i]) - 2.0 * w * self.vel[i];
            self.vel[i] += acc * dt;
            pos[i] += self.vel[i] * dt;
        }
        Some(*pos)
    }

    /// Paint the cloud centered at `c` (local window coordinates).
    pub fn paint(&self, p: &Painter, c: Pos2, t: f64) {
        let t = t as f32;
        let r = self.radius;
        let (cr, cg, cb) = COLOR;
        let rgba = |a: f32| Color32::from_rgba_unmultiplied(cr, cg, cb, (a.clamp(0.0, 1.0) * 255.0) as u8);

        // soft glow: stacked translucent discs approximate a radial gradient; it breathes slowly
        let breathe = 1.0 + 0.05 * (t * 0.9).sin();
        for k in 0..12 {
            let f = 1.0 - k as f32 / 12.0;
            p.circle_filled(c, r * 0.95 * f * breathe, rgba(0.018));
        }

        // slow ripples travelling outwards
        for k in 0..2 {
            let ph = ((t / 3.2) + k as f32 * 0.5).fract();
            let rr = r * (0.25 + 0.9 * ph);
            let a = 0.22 * (ph * (1.0 - ph) * 4.0).powi(2);
            p.circle_stroke(c, rr, Stroke::new(1.2, rgba(a)));
        }

        // particles: orbit slowly, wobble radially (a wave running outwards), twinkle
        for q in &self.particles {
            let ang = q.angle + q.speed * t + 0.25 * (t * 0.7 + q.phase).sin();
            let wave = 1.0 + 0.12 * (1.8 * t - q.radius * 7.0 + q.phase * 0.3).sin();
            let rad = r * q.radius * wave;
            let pos = c + egui::vec2(ang.cos(), ang.sin()) * rad;
            let fall = (-(q.radius * wave).powi(2) * 2.2).exp();
            let twinkle = 0.55 + 0.45 * (2.3 * t + q.phase * 3.0).sin();
            p.circle_filled(pos, q.size, rgba(0.55 * fall * twinkle));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaze::OneEuro;
    use rand::SeedableRng;

    /// Noisy gaze (±120 px jitter, like a webcam tracker) fixating one point, then jumping to another.
    #[test]
    fn calmer_than_the_ring() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (mut cloud, mut ring) = (GazeCloud::new(), OneEuro::new());
        let (mut cloud_moves, mut ring_moves) = (Vec::new(), Vec::new());
        let (mut last_c, mut last_r): (Option<[f64; 2]>, Option<[f64; 2]>) = (None, None);
        let dt = 1.0 / 30.0;
        let mut end = [0.0; 2];
        for i in 0..300 {
            let t = i as f64 * dt;
            let target = if i < 150 { (500.0, 400.0) } else { (1100.0, 600.0) };
            let g = (target.0 + rng.random_range(-120.0..120.0), target.1 + rng.random_range(-120.0..120.0));
            let c = cloud.update(t, dt, Some(g)).unwrap();
            let r = ring.filter([g.0, g.1], t);
            if let (Some(lc), Some(lr)) = (last_c, last_r) {
                if (30..150).contains(&i) {
                    // steady fixation: how much does the indicator wander per frame?
                    cloud_moves.push(((c[0] - lc[0]).powi(2) + (c[1] - lc[1]).powi(2)).sqrt());
                    ring_moves.push(((r[0] - lr[0]).powi(2) + (r[1] - lr[1]).powi(2)).sqrt());
                }
            }
            last_c = Some(c);
            last_r = Some(r);
            end = c;
        }
        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (cm, rm) = (avg(&cloud_moves), avg(&ring_moves));
        println!("mean per-frame motion during fixation: cloud {cm:.1}px, ring {rm:.1}px");
        assert!(cm < rm / 3.0, "cloud {cm:.1} vs ring {rm:.1}");
        // still follows real gaze shifts: settled near the second target after 5 s
        assert!((end[0] - 1100.0).abs() < 60.0 && (end[1] - 600.0).abs() < 60.0, "ended at {end:?}");
    }
}
