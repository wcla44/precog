//! Gaze smudge: a fuzzy, see-through-in-the-middle haze around where the user looks.
//!
//! It is rendered as a small procedural texture (animated fractal noise + grain), upscaled with linear filtering,
//! so it has no hard edge and never reads as a circle. Position comes straight from the fixation filter: it jumps
//! with saccades (no lag) and stays put during fixations. Its size follows the estimated gaze noise.

use egui::{Color32, ColorImage, Painter, Pos2, Rect, TextureHandle, TextureOptions};

/// Viewport side in points: holds the largest smudge.
pub const SIZE: f32 = 640.0;
/// Texture resolution (upscaled: the low resolution is what makes it soft).
const RES: usize = 96;
/// Texture covers [-EXTENT, EXTENT] in units of the smudge radius.
const EXTENT: f32 = 1.35;
const COLOR: [u8; 3] = [160, 205, 255];
const MAX_ALPHA: f32 = 0.42;

pub struct Smudge {
    tex: Option<TextureHandle>,
    radius: f32,
    frame: u32,
}

fn hash(x: i32, y: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    smooth(((x - a) / (b - a)).clamp(0.0, 1.0))
}

/// Value noise in [0, 1], seeded per octave.
fn noise(x: f32, y: f32, seed: i32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (smooth(x - x.floor()), smooth(y - y.floor()));
    let a = hash(xi, yi, seed) + (hash(xi + 1, yi, seed) - hash(xi, yi, seed)) * fx;
    let b = hash(xi, yi + 1, seed) + (hash(xi + 1, yi + 1, seed) - hash(xi, yi + 1, seed)) * fx;
    a + (b - a) * fy
}

fn fbm(x: f32, y: f32) -> f32 {
    noise(x, y, 1) * 0.55 + noise(x * 2.1, y * 2.1, 2) * 0.3 + noise(x * 4.3, y * 4.3, 3) * 0.15
}

impl Smudge {
    pub fn new() -> Self {
        Self { tex: None, radius: 120.0, frame: 0 }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Set the size directly (px): the halo's radius, chosen with the left hand.
    pub fn set_radius(&mut self, r: f32) {
        self.radius = r;
    }

    /// Ease the size towards the one implied by the gaze noise (px). Size changes slowly; position never lags.
    pub fn set_noise(&mut self, jitter: f64, dt: f64) {
        let target = (1.4 * jitter as f32 + 50.0).clamp(80.0, 230.0);
        self.radius += (target - self.radius) * (1.0 - (-dt as f32 * 2.0).exp());
    }

    fn render(&self, t: f32) -> ColorImage {
        let mut img = ColorImage::new([RES, RES], vec![Color32::TRANSPARENT; RES * RES]);
        for j in 0..RES {
            for i in 0..RES {
                let u = (i as f32 + 0.5) / RES as f32 * 2.0 * EXTENT - EXTENT;
                let v = (j as f32 + 0.5) / RES as f32 * 2.0 * EXTENT - EXTENT;
                // domain-warped noise drifting in two directions: churning, wave-like energy
                let wx = fbm(u * 1.6 + t * 0.21, v * 1.6 - t * 0.17);
                let wy = fbm(u * 1.6 - t * 0.19 + 5.2, v * 1.6 + t * 0.23 + 1.3);
                let n = fbm(u * 2.4 + wx * 1.8 + t * 0.12, v * 2.4 + wy * 1.8 - t * 0.1);
                let r = (u * u + v * v).sqrt();
                let inner = smoothstep(0.08, 0.5, r); // clear centre: you see what you look at
                let edge = r + 0.45 * (n - 0.5); // ragged, noisy outer edge: a smudge, not a disc
                let outer = 1.0 - smoothstep(0.45, 1.2, edge);
                let grain = 0.8 + 0.4 * hash(i as i32, j as i32, self.frame as i32 / 2); // particulate shimmer
                let a = MAX_ALPHA * inner * outer * (0.35 + 0.65 * n) * grain;
                img.pixels[j * RES + i] = Color32::from_rgba_unmultiplied(COLOR[0], COLOR[1], COLOR[2], (a.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
        img
    }

    /// Paint centred at `c` (window-local coordinates).
    pub fn paint(&mut self, ctx: &egui::Context, p: &Painter, c: Pos2, t: f64) {
        self.frame = self.frame.wrapping_add(1);
        let img = self.render(t as f32);
        let tex = self.tex.get_or_insert_with(|| ctx.load_texture("gaze-smudge", img.clone(), TextureOptions::LINEAR));
        tex.set(img, TextureOptions::LINEAR);
        let half = self.radius * EXTENT;
        let rect = Rect::from_center_size(c, egui::vec2(2.0 * half, 2.0 * half));
        p.image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    }
}
