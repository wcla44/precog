//! Gaze halo: a soft spot that stays put while the gaze wanders inside it and glides to a new spot when the gaze
//! settles outside. The left hand's openness sets its size; holding the left hand still for a second freezes it
//! (position and size); a right-hand swipe releases it.

use std::collections::VecDeque;

pub struct Halo {
    pub center: Option<(f64, f64)>,
    target: (f64, f64),
    pub radius: f64,
    /// The left hand has set the size; until then (and after a reset) it follows the gaze noise.
    pub sized_by_hand: bool,
    pub frozen: bool,
    /// Gaze samples since it left the halo: (since, sum x, sum y, n).
    outside: Option<(f64, f64, f64, u32)>,
    /// Left hand: (t, wrist, openness) over the last HOLD seconds.
    trail: VecDeque<(f64, [f32; 2], f32)>,
    /// A still left hand may freeze the halo. Cleared when it does (and on release), set again once the hand moves,
    /// so a hand that just stays still doesn't freeze it again right away.
    armed: bool,
}

/// Left hand as seen by the camera: wrist position (frame-height units) and openness (0 = fist, 1 = open).
#[derive(Clone, Copy, Debug)]
pub struct LeftHand {
    pub wrist: [f32; 2],
    pub openness: f32,
    /// Held up, fingers pointing up: only a raised hand can freeze the halo (not one resting on the desk).
    pub raised: bool,
}

impl Halo {
    pub const MIN_R: f64 = 40.0;
    pub const MAX_R: f64 = 400.0;
    const DEFAULT_R: f64 = 120.0;
    /// Time constant of the glide to a new spot, and of radius changes (s).
    const GLIDE: f64 = 0.18;
    const RESIZE: f64 = 0.15;
    /// The gaze must stay outside the halo this long before it moves (ignores stray saccades and noise).
    const OUTSIDE: f64 = 0.15;
    /// Gaze noise is ~60-100 px: a small halo still needs the gaze this far away before it moves.
    const MIN_MOVE: f64 = 90.0;
    /// Left hand held still this long freezes the halo.
    const HOLD: f64 = 1.0;
    const STILL_MOVE: f32 = 0.035; // wrist travel, frame-height units
    const STILL_OPEN: f32 = 0.08; // openness range

    pub fn new() -> Self {
        Self {
            center: None,
            target: (0.0, 0.0),
            radius: Self::DEFAULT_R,
            sized_by_hand: false,
            frozen: false,
            outside: None,
            trail: VecDeque::new(),
            armed: true,
        }
    }

    pub fn follow(&mut self, t: f64, gaze: Option<(f64, f64)>, dt: f64) {
        if self.frozen {
            return;
        }
        let Some(g) = gaze else {
            self.outside = None;
            return;
        };
        let Some(c) = self.center else {
            self.center = Some(g);
            self.target = g;
            return;
        };
        if dist(g, self.target) > self.radius.max(Self::MIN_MOVE) {
            let (since, sx, sy, n) = self.outside.unwrap_or((t, 0.0, 0.0, 0));
            let (sx, sy, n) = (sx + g.0, sy + g.1, n + 1);
            if t - since >= Self::OUTSIDE {
                self.target = (sx / n as f64, sy / n as f64);
                self.outside = None;
            } else {
                self.outside = Some((since, sx, sy, n));
            }
        } else {
            self.outside = None;
        }
        let a = 1.0 - (-dt / Self::GLIDE).exp();
        self.center = Some((c.0 + a * (self.target.0 - c.0), c.1 + a * (self.target.1 - c.1)));
    }

    /// Returns true when the halo just froze.
    pub fn left_hand(&mut self, t: f64, hand: Option<LeftHand>, dt: f64) -> bool {
        let Some(h) = hand else {
            self.trail.clear();
            self.armed = true;
            return false;
        };
        if !self.frozen {
            let want = Self::MIN_R + h.openness.clamp(0.0, 1.0) as f64 * (Self::MAX_R - Self::MIN_R);
            self.radius += (want - self.radius) * (1.0 - (-dt / Self::RESIZE).exp());
            self.sized_by_hand = true;
        }
        if !h.raised {
            // a lowered hand can't freeze: raising it and holding it still can
            self.trail.clear();
            self.armed = true;
            return false;
        }
        self.trail.push_back((t, h.wrist, h.openness));
        while self.trail.front().is_some_and(|p| t - p.0 > Self::HOLD) {
            self.trail.pop_front();
        }
        match self.still(t) {
            None => {} // not watched long enough yet
            Some(false) => self.armed = true,
            Some(true) if self.armed && !self.frozen && self.center.is_some() => {
                self.frozen = true;
                self.armed = false;
                return true;
            }
            Some(true) => {}
        }
        false
    }

    /// Still gliding towards a new spot (draw at a higher frame rate meanwhile).
    pub fn moving(&self) -> bool {
        self.center.is_some_and(|c| !self.frozen && dist(c, self.target) > 1.0)
    }

    /// Back to the defaults: released, default size, snaps to the next gaze point.
    pub fn reset(&mut self) {
        *self = Self::new();
        self.armed = false; // a left hand held still must not freeze it again right away
    }

    /// While the raised left hand is held still: how far the freeze is (0..1), for the gesture panel.
    pub fn freeze_progress(&self, t: f64) -> Option<f32> {
        if self.frozen || !self.armed {
            return None;
        }
        // the longest still stretch ending now
        let (mut x0, mut x1, mut y0, mut y1, mut o0, mut o1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        let mut since = None;
        for &(pt, w, o) in self.trail.iter().rev() {
            (x0, x1, y0, y1) = (x0.min(w[0]), x1.max(w[0]), y0.min(w[1]), y1.max(w[1]));
            (o0, o1) = (o0.min(o), o1.max(o));
            if (x1 - x0).hypot(y1 - y0) >= Self::STILL_MOVE || o1 - o0 >= Self::STILL_OPEN {
                break;
            }
            since = Some(pt);
        }
        since.map(|s| ((t - s) / Self::HOLD).clamp(0.0, 1.0) as f32)
    }

    /// Freeze where it is now (keyboard shortcut). False if there is no halo yet.
    pub fn freeze(&mut self) -> bool {
        if self.center.is_none() {
            return false;
        }
        self.frozen = true;
        self.armed = false;
        true
    }

    pub fn release(&mut self) {
        self.frozen = false;
        self.armed = false;
        self.trail.clear();
    }

    /// None until the hand has been watched for about HOLD seconds.
    fn still(&self, t: f64) -> Option<bool> {
        let first = self.trail.front()?;
        if t - first.0 < Self::HOLD * 0.9 {
            return None;
        }
        let (mut x0, mut x1, mut y0, mut y1, mut o0, mut o1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(_, w, o) in &self.trail {
            (x0, x1, y0, y1) = (x0.min(w[0]), x1.max(w[0]), y0.min(w[1]), y1.max(w[1]));
            (o0, o1) = (o0.min(o), o1.max(o));
        }
        Some((x1 - x0).hypot(y1 - y0) < Self::STILL_MOVE && o1 - o0 < Self::STILL_OPEN)
    }
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_small_moves_and_follows_big_ones() {
        let mut h = Halo::new();
        let dt = 1.0 / 30.0;
        let mut t = 0.0;
        for i in 0..60 {
            t += dt;
            h.follow(t, Some((500.0 + if i % 2 == 0 { 40.0 } else { -40.0 }, 500.0)), dt);
        }
        let c = h.center.unwrap();
        assert!((c.0 - 540.0).abs() < 1.0, "jitter inside the halo must not move it: {c:?}");
        for _ in 0..60 {
            t += dt;
            h.follow(t, Some((1200.0, 500.0)), dt);
        }
        assert!((h.center.unwrap().0 - 1200.0).abs() < 5.0);
    }

    #[test]
    fn still_left_hand_freezes_once() {
        let mut h = Halo::new();
        h.follow(0.0, Some((100.0, 100.0)), 0.03);
        let still = LeftHand { wrist: [0.5, 0.5], openness: 0.5, raised: true };
        let mut froze = 0;
        for i in 0..90 {
            froze += h.left_hand(i as f64 / 30.0, Some(still), 1.0 / 30.0) as u32;
        }
        assert_eq!(froze, 1);
        assert!(h.frozen);
        h.release();
        for i in 90..180 {
            froze += h.left_hand(i as f64 / 30.0, Some(still), 1.0 / 30.0) as u32;
        }
        assert_eq!(froze, 1, "a hand that never moved must not freeze it again after release");
    }

    #[test]
    fn freeze_progress_fills_while_still_and_resets_on_motion() {
        let mut h = Halo::new();
        h.follow(0.0, Some((100.0, 100.0)), 0.03);
        let hand = |x: f32| Some(LeftHand { wrist: [x, 0.5], openness: 0.5, raised: true });
        for i in 0..15 {
            h.left_hand(i as f64 / 30.0, hand(0.5), 1.0 / 30.0);
        }
        let p = h.freeze_progress(14.0 / 30.0).unwrap();
        assert!((0.4..0.6).contains(&p), "half a second still: {p}");
        h.left_hand(15.0 / 30.0, hand(0.7), 1.0 / 30.0); // moved
        assert!(h.freeze_progress(15.0 / 30.0).unwrap_or(0.0) < 0.1);
    }

    #[test]
    fn resting_hand_never_freezes() {
        let mut h = Halo::new();
        h.follow(0.0, Some((100.0, 100.0)), 0.03);
        let resting = LeftHand { wrist: [0.6, 0.97], openness: 0.2, raised: false };
        let froze = (0..90).filter(|&i| h.left_hand(i as f64 / 30.0, Some(resting), 1.0 / 30.0)).count();
        assert_eq!(froze, 0);
        assert!(!h.frozen);
    }
}
