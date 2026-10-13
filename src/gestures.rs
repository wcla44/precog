//! Hand pose classification + swipe detection from 21 hand landmarks (no LLM involved).

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pose {
    Open,
    Fist,
    Pinch,
    Other,
}

impl Pose {
    pub fn icon(p: Option<Pose>) -> &'static str {
        match p {
            Some(Pose::Open) => "✋",
            Some(Pose::Fist) => "✊",
            Some(Pose::Pinch) => "🤏",
            Some(Pose::Other) => "🖐",
            None => "·",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swipe {
    Left,
    Right,
}

const WRIST: usize = 0;
const THUMB_TIP: usize = 4;
const INDEX_TIP: usize = 8;
const FINGERS: [(usize, usize, usize); 4] = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]; // (mcp, pip, tip)

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// `lm`: 21 points in any isotropic unit (pixels).
pub fn classify(lm: &[[f32; 2]]) -> Pose {
    let size = dist(lm[WRIST], lm[9]).max(1e-6);
    let extended: Vec<bool> =
        FINGERS.iter().map(|&(_, pip, tip)| dist(lm[tip], lm[WRIST]) > dist(lm[pip], lm[WRIST]) * 1.15).collect();
    let curled = FINGERS.iter().filter(|&&(mcp, _, tip)| dist(lm[tip], lm[WRIST]) < dist(lm[mcp], lm[WRIST]) * 1.1).count();
    let thumb_index = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / size;
    let thumb_out = dist(lm[THUMB_TIP], lm[5]) / size > 0.55;
    if curled >= 4 {
        Pose::Fist
    } else if thumb_index < 0.25 && extended[1..].iter().any(|&e| e) {
        Pose::Pinch
    } else if extended.iter().all(|&e| e) && thumb_out {
        Pose::Open
    } else {
        Pose::Other
    }
}

/// Swipes in the USER's frame of reference: Left = the user moves the hand to their left.
pub struct SwipeDetector {
    trail: VecDeque<(f64, f32, f32)>,
    last: f64,
    pub mirrored: bool,
}

impl SwipeDetector {
    const WINDOW: f64 = 0.45;
    const MIN_DX: f32 = 0.30; // in hand-size-independent frame units (height = 1)
    const MAX_DY: f32 = 0.18;
    const COOLDOWN: f64 = 1.0;

    pub fn new() -> Self {
        Self { trail: VecDeque::new(), last: -1e9, mirrored: false }
    }

    /// `wrist`: normalised by frame height (x in [0, w/h]).
    pub fn update(&mut self, t: f64, wrist: Option<[f32; 2]>) -> Option<Swipe> {
        let Some(w) = wrist else {
            self.trail.clear();
            return None;
        };
        self.trail.push_back((t, w[0], w[1]));
        while self.trail.front().is_some_and(|p| t - p.0 > Self::WINDOW) {
            self.trail.pop_front();
        }
        if t - self.last < Self::COOLDOWN || self.trail.len() < 3 {
            return None;
        }
        let (first, last) = (self.trail.front()?, self.trail.back()?);
        // raw webcam frames are not mirrored: the user's left is the image's right (+x)
        let sign = if self.mirrored { -1.0 } else { 1.0 };
        let (dx, dy) = ((last.1 - first.1) * sign, last.2 - first.2);
        if dx.abs() > Self::MIN_DX && dy.abs() < Self::MAX_DY {
            self.last = t;
            self.trail.clear();
            return Some(if dx > 0.0 { Swipe::Left } else { Swipe::Right });
        }
        None
    }
}

/// A pose only counts once held for `hold` seconds.
pub struct Debounce {
    hold: f64,
    pose: Option<Pose>,
    since: f64,
    stable: Option<Pose>,
}

impl Debounce {
    pub fn new(hold: f64) -> Self {
        Self { hold, pose: None, since: 0.0, stable: None }
    }

    pub fn update(&mut self, t: f64, pose: Option<Pose>) -> Option<Pose> {
        if pose != self.pose {
            self.pose = pose;
            self.since = t;
        }
        if t - self.since >= self.hold {
            self.stable = pose;
        }
        self.stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(ext: [bool; 4], pinch: bool) -> Vec<[f32; 2]> {
        let mut lm = vec![[0.0f32; 2]; 21];
        lm[0] = [0.5, 0.8];
        for (i, &(mcp, pip, tip)) in FINGERS.iter().enumerate() {
            let x = 0.42 + i as f32 * 0.05;
            lm[mcp] = [x, 0.6];
            lm[pip] = [x, 0.5];
            lm[pip + 1] = [x, 0.45];
            lm[tip] = if ext[i] { [x, 0.38] } else { [x, 0.62] };
        }
        lm[1..4].copy_from_slice(&[[0.4, 0.7]; 3]);
        lm[4] = if pinch { lm[8] } else { [0.30, 0.62] };
        lm
    }

    #[test]
    fn poses() {
        assert_eq!(classify(&hand([true; 4], false)), Pose::Open);
        assert_eq!(classify(&hand([false; 4], false)), Pose::Fist);
        assert_eq!(classify(&hand([true; 4], true)), Pose::Pinch);
    }

    #[test]
    fn swipe() {
        let mut s = SwipeDetector::new();
        let got: Vec<_> = (0..7).map(|t| s.update(t as f64 / 15.0, Some([0.5 + t as f32 * 0.08, 0.5]))).collect();
        assert!(got.contains(&Some(Swipe::Left)));
    }
}
