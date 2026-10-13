//! Geometric gaze model: where the eyes are, where the screens are, and which way the eyes look.
//!
//! A regression straight from eye features to screen pixels only holds near the head position it was calibrated at
//! (moving on the chair breaks it). Here the problem is split, as in model-based eye trackers:
//! - the eye position (mm) and head rotation come from the face mesh (PnP on a metric 3D face model, see
//!   `unigaze::head_and_eyes`);
//! - the screens are placed in 3D in the camera frame. A laptop's webcam sits in its screen bezel, so that screen is
//!   known whatever the lid angle: the plane z = 0, just below the camera. The other screens' poses are estimated at
//!   calibration, starting from the system layout and physical sizes (`layout`);
//! - a per-user ridge regression maps eye features to the gaze direction in the head frame (`gaze::GazeModel`);
//! - the gaze point is where that ray, from the eyes, meets a screen.
//!
//! Camera frame: x to the right of the image, y down, z from the camera towards the user. The image is not mirrored,
//! so the user's right (screen +x) is the camera's -x.

use crate::layout::Physical;
use crate::monitors::Monitor;
use nalgebra::{Rotation3, Vector3};
use serde::{Deserialize, Serialize};

/// Feature columns (see `tracker::features`): head rotation vector, eye position (mm).
const ROT: usize = 14;
const EYE: usize = 17;
/// The webcam's centre is this far above the top edge of the laptop panel (mm).
const CAMERA_ABOVE_SCREEN: f64 = 8.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Screen3 {
    /// Logical rect (left, top, w, h) and physical size (w, h) in mm.
    pub px: (i32, i32, u32, u32),
    pub mm: (f64, f64),
    /// Top-left corner, and unit vectors along +x and +y pixels, in the camera frame.
    pub origin: [f64; 3],
    pub ex: [f64; 3],
    pub ey: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geo {
    pub screens: Vec<Screen3>,
    /// Corrects the eye distance from PnP for the unknown camera focal length.
    pub depth_scale: f64,
    /// Per non-primary screen: offset (right, down, towards the user; mm) and tilt (rad) from its starting pose.
    pub params: Vec<[f64; 4]>,
    /// Human-readable arrangement, for messages.
    pub label: String,
}

fn v(a: [f64; 3]) -> Vector3<f64> {
    Vector3::new(a[0], a[1], a[2])
}

/// Gaze direction in the head frame <-> (yaw, pitch). Looking straight at the camera from in front of it is (0, 0).
fn to_angles(d: Vector3<f64>) -> [f64; 2] {
    let d = d.normalize();
    [d.x.atan2(-d.z), d.y.clamp(-1.0, 1.0).asin()]
}

fn from_angles(a: [f64; 2]) -> Vector3<f64> {
    let (yaw, pitch) = (a[0], a[1]);
    Vector3::new(pitch.cos() * yaw.sin(), pitch.sin(), -pitch.cos() * yaw.cos())
}

impl Geo {
    /// Screens from a planar physical arrangement: the primary one under the camera (z = 0), the others in the same
    /// plane at their planar positions, then moved by `params`.
    pub fn new(phys: &Physical, mons: &[Monitor], depth_scale: f64, params: Vec<[f64; 4]>) -> Geo {
        let pw = phys.screens[0].1 .2;
        let screens = phys
            .screens
            .iter()
            .zip(mons)
            .enumerate()
            .map(|(i, (&(px, (x, y, w, h)), _))| {
                // planar (x right, y down; primary's top-left at 0,0) -> camera frame
                let mut origin = Vector3::new(pw / 2.0 - x, CAMERA_ABOVE_SCREEN + y, 0.0);
                let ex = Vector3::new(-1.0, 0.0, 0.0);
                let mut ey = Vector3::new(0.0, 1.0, 0.0);
                if i > 0 {
                    let p = params.get(i - 1).copied().unwrap_or([0.0; 4]);
                    origin += Vector3::new(-p[0], p[1], p[2]);
                    // tilt about the screen's horizontal axis: positive = top further from the user
                    ey = Vector3::new(0.0, p[3].cos(), p[3].sin());
                }
                Screen3 { px, mm: (w, h), origin: origin.into(), ex: ex.into(), ey: ey.into() }
            })
            .collect();
        Geo { screens, depth_scale, params, label: phys.label.clone() }
    }

    fn eye(&self, f: &[f64]) -> Vector3<f64> {
        Vector3::new(f[EYE], f[EYE + 1], f[EYE + 2] * self.depth_scale)
    }

    fn head(f: &[f64]) -> Rotation3<f64> {
        Rotation3::new(Vector3::new(f[ROT], f[ROT + 1], f[ROT + 2]))
    }

    /// 3D point (camera frame) of a logical pixel, on the screen that holds it (or the nearest one).
    pub fn point(&self, x: f64, y: f64) -> Vector3<f64> {
        let s = self
            .screens
            .iter()
            .min_by(|a, b| out_px(a, x, y).total_cmp(&out_px(b, x, y)))
            .expect("at least one screen");
        let (u, w) = ((x - s.px.0 as f64) / s.px.2 as f64 * s.mm.0, (y - s.px.1 as f64) / s.px.3 as f64 * s.mm.1);
        v(s.origin) + v(s.ex) * u + v(s.ey) * w
    }

    /// Head-frame gaze angles that would put the gaze on the logical pixel (x, y), for that frame's head pose.
    pub fn angles_to(&self, f: &[f64], x: f64, y: f64) -> [f64; 2] {
        let d = self.point(x, y) - self.eye(f);
        to_angles(Self::head(f).inverse() * d)
    }

    /// Where the gaze ray (head-frame angles) from the eyes meets the screens, in logical pixels. A ray that misses
    /// every screen lands on the nearest one, extrapolated beyond its edge.
    pub fn hit(&self, f: &[f64], angles: [f64; 2]) -> Option<(f64, f64)> {
        let (o, d) = (self.eye(f), Self::head(f) * from_angles(angles));
        let mut best: Option<(f64, f64, f64)> = None;
        for s in &self.screens {
            let (so, ex, ey) = (v(s.origin), v(s.ex), v(s.ey));
            let n = ex.cross(&ey);
            let den = n.dot(&d);
            if den.abs() < 1e-9 {
                continue;
            }
            let k = n.dot(&(so - o)) / den;
            if k <= 0.0 {
                continue;
            }
            let p = o + d * k - so;
            let (u, w) = (p.dot(&ex), p.dot(&ey));
            let out = (-u).max(u - s.mm.0).max(0.0).hypot((-w).max(w - s.mm.1).max(0.0));
            let px = (s.px.0 as f64 + u / s.mm.0 * s.px.2 as f64, s.px.1 as f64 + w / s.mm.1 * s.px.3 as f64);
            if best.is_none_or(|b| out < b.2) {
                best = Some((px.0, px.1, out));
            }
        }
        best.map(|b| (b.0, b.1))
    }
}

impl Geo {
    /// The same screens in a new screen setup (screens plugged / unplugged / moved in the system layout). Screens seen
    /// before (same resolution and physical size) keep their 3D pose, wherever the system now places them; new ones
    /// get a starting pose from the new layout. Returns the geometry and the new screens (to calibrate).
    pub fn adapt(&self, mons: &[Monitor]) -> (Geo, Vec<Monitor>) {
        let cam = crate::monitors::camera_first(mons);
        let prior = Geo::new(&Physical::candidates(&cam)[0], &cam, self.depth_scale, Vec::new());
        let mut used = vec![false; self.screens.len()];
        let mut new_screens = Vec::new();
        let screens = cam
            .iter()
            .zip(&prior.screens)
            .map(|(m, p)| {
                let mm = m.mm.unwrap_or((m.width as f64 * 0.25, m.height as f64 * 0.25));
                let same = |s: &Screen3| s.px.2 == m.width && s.px.3 == m.height && (s.mm.0 - mm.0).abs() < 3.0 && (s.mm.1 - mm.1).abs() < 3.0;
                match self.screens.iter().enumerate().find(|(i, s)| !used[*i] && same(s)) {
                    Some((i, s)) => {
                        used[i] = true;
                        Screen3 { px: m.layout(), ..s.clone() }
                    }
                    None => {
                        new_screens.push(*m);
                        p.clone()
                    }
                }
            })
            .collect();
        (Geo { screens, depth_scale: self.depth_scale, params: self.params.clone(), label: self.label.clone() }, new_screens)
    }

    /// Logical pixel of a 3D point lying on one of the screens (within 5 mm), if any.
    pub fn locate(&self, p: Vector3<f64>) -> Option<(f64, f64)> {
        self.screens.iter().find_map(|s| {
            let (so, ex, ey) = (v(s.origin), v(s.ex), v(s.ey));
            let d = p - so;
            if ex.cross(&ey).dot(&d).abs() > 5.0 {
                return None;
            }
            let (u, w) = (d.dot(&ex), d.dot(&ey));
            ((-5.0..=s.mm.0 + 5.0).contains(&u) && (-5.0..=s.mm.1 + 5.0).contains(&w))
                .then(|| (s.px.0 as f64 + u / s.mm.0 * s.px.2 as f64, s.px.1 as f64 + w / s.mm.1 * s.px.3 as f64))
        })
    }
}

/// How far (px) a logical point is outside a screen's rect (0 inside).
fn out_px(s: &Screen3, x: f64, y: f64) -> f64 {
    let (l, t, w, h) = (s.px.0 as f64, s.px.1 as f64, s.px.2 as f64, s.px.3 as f64);
    (l - x).max(x - (l + w)).max(0.0).hypot((t - y).max(y - (t + h)).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Vec<Monitor>, Geo) {
        let mons = vec![
            Monitor { left: 0, top: 1440, width: 1920, height: 1080, primary: true, mm: Some((294.0, 165.0)), builtin: true },
            Monitor { left: 0, top: 0, width: 2560, height: 1440, primary: false, mm: Some((597.0, 336.0)), builtin: false },
        ];
        let phys = Physical::candidates(&mons).remove(0);
        let geo = Geo::new(&phys, &mons, 1.0, vec![[30.0, -40.0, -120.0, 0.2]]);
        (mons, geo)
    }

    #[test]
    fn rays_land_where_they_were_aimed() {
        let (_, geo) = setup();
        // eye 600 mm in front of the camera, slightly left and below, head turned a little
        let mut f = vec![0.0; crate::gaze::NF];
        let rv = Vector3::new(0.1, -0.15, 0.05);
        f[ROT..ROT + 3].copy_from_slice(rv.as_slice());
        f[EYE..EYE + 3].copy_from_slice(&[40.0, 60.0, 600.0]);
        for &(x, y) in &[(100.0, 1500.0), (1800.0, 2400.0), (960.0, 1980.0), (50.0, 30.0), (2500.0, 1400.0), (1280.0, 700.0)] {
            let a = geo.angles_to(&f, x, y);
            let (hx, hy) = geo.hit(&f, a).unwrap();
            assert!((hx - x).abs() < 1e-6 && (hy - y).abs() < 1e-6, "({x},{y}) -> ({hx},{hy})");
        }
    }

    #[test]
    fn adapts_to_unplugged_and_moved_screens() {
        let (mons, geo) = setup();
        // external screen unplugged: the laptop alone, now at the origin of the layout
        let alone = vec![Monitor { left: 0, top: 0, ..mons[0] }];
        let (g1, new) = geo.adapt(&alone);
        assert!(new.is_empty());
        assert_eq!(g1.screens.len(), 1);
        // the same physical spot on the laptop panel, at its new logical position
        let p = geo.point(500.0, 1440.0 + 300.0);
        let (x, y) = g1.locate(p).unwrap();
        assert!((x - 500.0).abs() < 1e-6 && (y - 300.0).abs() < 1e-6);
        // plugged back, configured to the right this time: it keeps its physical pose (still above the laptop)
        let right = vec![Monitor { left: 0, top: 0, ..mons[0] }, Monitor { left: 1920, top: 0, ..mons[1] }];
        let (g2, new) = geo.adapt(&right);
        assert!(new.is_empty());
        let q = geo.point(1280.0, 700.0); // a spot on the external screen, old layout
        let (x, y) = g2.locate(q).unwrap();
        assert!((x - (1920.0 + 1280.0)).abs() < 1e-6 && (y - 700.0).abs() < 1e-6);
        // a screen never seen: to calibrate
        let other = vec![mons[0], Monitor { width: 1920, height: 1200, mm: Some((518.0, 324.0)), ..mons[1] }];
        assert_eq!(geo.adapt(&other).1.len(), 1);
    }

    #[test]
    fn laptop_screen_is_under_the_camera_and_mirrored() {
        let (_, geo) = setup();
        // top-left pixel of the laptop screen: the user's left = the camera's +x
        let p = geo.point(0.0, 1440.0);
        assert!((p.x - 147.0).abs() < 1e-9 && (p.y - CAMERA_ABOVE_SCREEN).abs() < 1e-9 && p.z == 0.0);
        let p = geo.point(1920.0, 2520.0);
        assert!((p.x + 147.0).abs() < 1e-9 && (p.y - CAMERA_ABOVE_SCREEN - 165.0).abs() < 1e-9);
    }
}
