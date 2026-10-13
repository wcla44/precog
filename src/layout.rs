//! Physical arrangement of the screens, in millimetres.
//!
//! Gaze is a smooth function of eye and head pose in physical space, not in pixels: screens have different pixel
//! densities, and the system settings may place a screen where it isn't (a monitor configured to the right of the
//! laptop while it actually stands above it). So the gaze models are fitted in this space. The system layout and
//! the screens' physical sizes are only the starting point: after a calibration, the plausible arrangements are
//! tried and the one that explains the gaze best is kept (see `candidates`).

use crate::monitors::Monitor;
use serde::{Deserialize, Serialize};

/// Assumed when a screen doesn't report its physical size (projectors, some VMs): ~100 dpi.
const DEFAULT_MM_PER_PX: f64 = 0.25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Physical {
    /// Per monitor, in the order of `monitors::list()`: logical rect (left, top, w, h) and physical rect
    /// (x, y, w, h) in mm.
    pub screens: Vec<((i32, i32, u32, u32), (f64, f64, f64, f64))>,
    /// How the non-primary screens were placed, for messages ("DP-1: above, centred").
    pub label: String,
}

fn size_mm(m: &Monitor) -> (f64, f64) {
    match m.mm {
        Some((w, h)) if w > 0.0 && h > 0.0 => (w, h),
        _ => (m.width as f64 * DEFAULT_MM_PER_PX, m.height as f64 * DEFAULT_MM_PER_PX),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Above,
    Below,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Align {
    Start,
    Centre,
    End,
    /// Same offset as in the system layout (converted to mm with the primary screen's density).
    System,
}

impl Physical {
    /// Logical pixel -> mm, through the screen that contains the point (or the nearest one, extrapolated).
    pub fn to_mm(&self, x: f64, y: f64) -> (f64, f64) {
        let (l, p) = self.nearest(|(l, _)| dist_rect(x, y, l.0 as f64, l.1 as f64, l.2 as f64, l.3 as f64));
        (p.0 + (x - l.0 as f64) * p.2 / l.2 as f64, p.1 + (y - l.1 as f64) * p.3 / l.3 as f64)
    }

    /// mm -> logical pixel, through the screen that contains the point (or the nearest one, extrapolated).
    pub fn to_px(&self, x: f64, y: f64) -> (f64, f64) {
        let (l, p) = self.nearest(|(_, p)| dist_rect(x, y, p.0, p.1, p.2, p.3));
        (l.0 as f64 + (x - p.0) * l.2 as f64 / p.2, l.1 as f64 + (y - p.1) * l.3 as f64 / p.3)
    }

    fn nearest(&self, d: impl Fn(&((i32, i32, u32, u32), (f64, f64, f64, f64))) -> f64) -> ((i32, i32, u32, u32), (f64, f64, f64, f64)) {
        *self.screens.iter().min_by(|a, b| d(a).total_cmp(&d(b))).expect("at least one screen")
    }

    /// Arrangements to try, the system's own first: each non-primary screen on each side of the primary one, at
    /// each alignment. With more than two screens, every other screen keeps its system placement while one varies.
    pub fn candidates(mons: &[Monitor]) -> Vec<Physical> {
        let system: Vec<(Side, Align)> = mons.iter().skip(1).map(|m| (system_side(&mons[0], m), Align::System)).collect();
        let mut out = vec![Self::arrange(mons, &system)];
        for i in 0..system.len() {
            for side in [Side::Above, Side::Below, Side::Left, Side::Right] {
                for align in [Align::Start, Align::Centre, Align::End] {
                    let mut placement = system.clone();
                    placement[i] = (side, align);
                    let c = Self::arrange(mons, &placement);
                    if !out.iter().any(|o| o.screens == c.screens) {
                        out.push(c);
                    }
                }
            }
        }
        out
    }

    /// Primary screen at the origin; the others next to it, without overlapping it.
    fn arrange(mons: &[Monitor], placement: &[(Side, Align)]) -> Physical {
        let p = &mons[0];
        let (pw, ph) = size_mm(p);
        let mm_per_px = (pw / p.width as f64, ph / p.height as f64);
        let mut screens = vec![(p.layout(), (0.0, 0.0, pw, ph))];
        let mut label = Vec::new();
        for (m, &(side, align)) in mons.iter().skip(1).zip(placement) {
            let (w, h) = size_mm(m);
            // offset along the shared edge
            let along = |len_p: f64, len_m: f64, sys_px: f64, k: f64| match align {
                Align::Start => 0.0,
                Align::Centre => (len_p - len_m) / 2.0,
                Align::End => len_p - len_m,
                Align::System => sys_px * k,
            };
            let (x, y) = match side {
                Side::Above => (along(pw, w, (m.left - p.left) as f64, mm_per_px.0), -h),
                Side::Below => (along(pw, w, (m.left - p.left) as f64, mm_per_px.0), ph),
                Side::Left => (-w, along(ph, h, (m.top - p.top) as f64, mm_per_px.1)),
                Side::Right => (pw, along(ph, h, (m.top - p.top) as f64, mm_per_px.1)),
            };
            screens.push((m.layout(), (x, y, w, h)));
            label.push(format!("{} {}", format!("{side:?}").to_lowercase(), match align {
                Align::Start => "start-aligned",
                Align::Centre => "centred",
                Align::End => "end-aligned",
                Align::System => "as configured",
            }));
        }
        Physical { screens, label: label.join(", ") }
    }
}

/// Side of the primary screen where the system layout puts `m`.
fn system_side(p: &Monitor, m: &Monitor) -> Side {
    let (pcx, pcy) = (p.left as f64 + p.width as f64 / 2.0, p.top as f64 + p.height as f64 / 2.0);
    let (mcx, mcy) = (m.left as f64 + m.width as f64 / 2.0, m.top as f64 + m.height as f64 / 2.0);
    let (dx, dy) = ((mcx - pcx) / (p.width + m.width) as f64, (mcy - pcy) / (p.height + m.height) as f64);
    match (dx.abs() > dy.abs(), dx > 0.0, dy > 0.0) {
        (true, true, _) => Side::Right,
        (true, false, _) => Side::Left,
        (false, _, true) => Side::Below,
        (false, _, false) => Side::Above,
    }
}

fn dist_rect(x: f64, y: f64, rx: f64, ry: f64, rw: f64, rh: f64) -> f64 {
    let dx = (rx - x).max(0.0).max(x - (rx + rw));
    let dy = (ry - y).max(0.0).max(y - (ry + rh));
    dx.hypot(dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mons() -> Vec<Monitor> {
        // laptop (primary) below a large monitor, both left-aligned in the system layout
        vec![
            Monitor { left: 0, top: 1440, width: 1920, height: 1080, primary: true, mm: Some((294.0, 165.0)), builtin: true },
            Monitor { left: 0, top: 0, width: 2560, height: 1440, primary: false, mm: Some((597.0, 336.0)), builtin: false },
        ]
    }

    #[test]
    fn system_layout_first_and_round_trip() {
        let c = Physical::candidates(&mons());
        assert_eq!(c[0].screens[1].1, (0.0, -336.0, 597.0, 336.0), "external screen above, as configured");
        assert!(c.len() > 10);
        for phys in &c {
            for &(x, y) in &[(10.0, 1450.0), (1900.0, 2500.0), (1280.0, 700.0), (2550.0, 5.0)] {
                let (mx, my) = phys.to_mm(x, y);
                let (px, py) = phys.to_px(mx, my);
                assert!((px - x).abs() < 1e-6 && (py - y).abs() < 1e-6, "{} ({x},{y})", phys.label);
            }
        }
        // centred: the 597 mm monitor overhangs the 294 mm laptop on both sides
        let centred = c.iter().find(|p| p.label == "above centred").unwrap();
        assert!((centred.screens[1].1 .0 - (294.0 - 597.0) / 2.0).abs() < 1e-9);
    }
}
