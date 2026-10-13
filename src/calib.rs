//! Calibration (inspired by WebGazer's click learning and by pursuit calibration, Pfeuffer et al. UIST 2013 /
//! Drewes et al. ETRA 2019), then a validation that decides which model is kept:
//!
//! 1. Laser dots: the mouse pointer becomes a red laser; the user brings it onto each dot while following it with the
//!    eyes, and clicks the dot. The click is a sure sample (gaze, pointer and dot coincide). The pointer's way there
//!    is used too, where the gaze follows the pointer (it often runs ahead to the target: those frames are dropped).
//! 2. Reading: short texts at different places (two corners per screen); the user reads them moving the laser under
//!    the words, and clicks at the end of each. Reading gives long horizontal sweeps over the whole screen, at
//!    realistic gaze behaviour; the eye may run a little ahead of the pointer (negative lag allowed), and only the
//!    frames where the gaze moves with the pointer are kept.
//! 3. Pursuit: the user follows a dot moving smoothly over every screen (~18 s, under ~10 deg/s). The predicted gaze
//!    is shown all along and the model is refitted every second, so the ring converges while the user watches. The
//!    labels are always the target, never the ring: frames where the gaze doesn't move with the target (a glance at
//!    the ring, a blink, looking away) are dropped by a windowed correlation test, which ignores a constant offset
//!    and so works with an imperfect model.
//! 4. Validation: fixed dots in the corners and centre of every screen, looked at for ~1.4 s, never learned from.
//!    The error is measured per screen, corners vs centre, in px and degrees. A new pursuit is kept only if the model
//!    with it does better on that validation than the best model so far.
//! 5. Test: the gaze is shown live with markers; the user decides: Y done, R one more pursuit, A start over.
//!
//! The user may move on the chair meanwhile: the geometric model (`geo`) uses the head position.
//! Quick recalibration (`quick`) keeps the simple dwell dots.

use crate::gaze::{GazeModel, NF};
use crate::monitors::Monitor;
use crate::tracker::Tracker;
use egui::{Align2, Color32, FontId, Painter, Pos2, Stroke};

const RED: Color32 = Color32::from_rgb(255, 59, 48);
/// A click this close to the dot (px) counts.
const HIT: f64 = 45.0;
const PURSUIT_S: f64 = 18.0;
/// Pursuit speed limits (mm/s on the screen: ~6-12 deg/s at 60 cm).
const SPEED: (f64, f64) = (60.0, 130.0);
/// Groups (cross-validation folds keep each group together): dots, the way to each dot, 1 s pursuit chunks.
const MOVE_GROUP: i64 = 200;
const PURSUIT_GROUP: i64 = 400;
const READ_GROUP: i64 = 300;
/// Short passages for the reading step (any text works: it is the reading that matters).
const TEXTS: [&str; 6] = [
    "Le soleil se levait à peine sur la vallée quand les premiers marcheurs ont quitté le refuge. Le sentier longeait le torrent, puis grimpait en lacets serrés vers le col, encore couvert de neige à cette saison.",
    "Une bonne interface ne se remarque pas : elle laisse l'attention sur la tâche. Chaque seconde passée à chercher un bouton est une seconde volée à ce que l'on voulait vraiment faire.",
    "Les phares de la côte bretonne ont longtemps été tenus par des gardiens qui vivaient seuls pendant des semaines, à guetter le moindre navire en difficulté au milieu des tempêtes d'hiver.",
    "Pour bien lire un graphique, regardez d'abord les axes et les unités, puis la tendance générale, et seulement ensuite les détails. Beaucoup d'erreurs viennent d'une échelle mal comprise.",
    "Le pain se travaille avec patience : une farine simple, de l'eau, du sel et du levain. La pâte repose toute la nuit, et c'est ce temps long qui lui donne sa croûte et son goût.",
    "Dans une forêt ancienne, les arbres communiquent par leurs racines et par un réseau de champignons. Ils échangent de l'eau, des sucres et même des signaux d'alerte quand un parasite approche.",
];
/// Validation dots: per screen, corners and centre, and how long each is shown.
/// Close to the edges (~50-60 px): the model has to learn where the screen really ends, or the gaze won't reach it.
const VAL_SPOTS: [(f64, f64); 5] = [(0.025, 0.035), (0.975, 0.035), (0.5, 0.5), (0.975, 0.965), (0.025, 0.965)];
const VAL_SETTLE: f64 = 0.6;
const VAL_COLLECT: f64 = 0.8;

#[derive(Clone, Copy)]
pub enum Next {
    Reading,
    Pursuit,
    Validation,
}

pub enum Stage {
    Intro,
    Laser { i: usize, since: f64 },
    ReadingIntro,
    Reading { i: usize, since: f64 },
    /// Model fitting runs on the next frame, after this screen was drawn.
    Fitting { next: Next, drawn: bool },
    PursuitIntro,
    Pursuit { t0: f64, next_refit: f64 },
    Validation { i: Option<usize>, t0: f64 },
    Review,
    Quick { i: Option<usize>, t0: f64, settle: f64 },
}

pub enum Outcome {
    Running,
    /// Done: the models hold the samples (full calibration: fitted, with geometry).
    Finished,
}

/// Smooth path through waypoints at constant physical speed.
struct Path {
    pts: Vec<(f64, f64, f64)>, // (x, y, cumulative mm)
    speed: f64,
}

impl Path {
    fn new(mons: &[Monitor]) -> Path {
        use rand::Rng;
        let mut rng = rand::rng();
        // per screen (top to bottom, then left to right): its four corner areas in a loop, then one inner point
        let mut order: Vec<&Monitor> = mons.iter().collect();
        order.sort_by_key(|m| (m.top, m.left));
        let mut way = Vec::new();
        // with few screens one loop is short (a single laptop screen: ~7 s): loop again, with other waypoints
        let loops = 2usize.div_ceil(order.len().max(1));
        for m in (0..loops).flat_map(|_| order.iter().copied()) {
            let mut j = |a: f64, b: f64| rng.random_range(a..b);
            for (fx, fy) in [(j(0.02, 0.15), j(0.03, 0.15)), (j(0.85, 0.98), j(0.03, 0.15)), (j(0.85, 0.98), j(0.85, 0.97)), (j(0.02, 0.15), j(0.85, 0.97)), (j(0.35, 0.65), j(0.35, 0.65))] {
                way.push((m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64));
            }
        }
        // Catmull-Rom through the waypoints, densified
        let mut pts: Vec<(f64, f64, f64)> = Vec::new();
        let at = |i: isize| way[i.clamp(0, way.len() as isize - 1) as usize];
        for i in 0..way.len() as isize - 1 {
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            for k in 0..40 {
                let t = k as f64 / 40.0;
                let (t2, t3) = (t * t, t * t * t);
                let c = |a: f64, b: f64, c: f64, d: f64| 0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (-a + 3.0 * b - 3.0 * c + d) * t3);
                // the spline overshoots its waypoints a little: keep it on the screens
                let (x, y) = crate::monitors::clamp(mons, c(p0.0, p1.0, p2.0, p3.0), c(p0.1, p1.1, p2.1, p3.1));
                let s = match pts.last() {
                    Some(&(px, py, ps)) => ps + (x - px).hypot(y - py) * mm_per_px(mons, x, y),
                    None => 0.0,
                };
                pts.push((x, y, s));
            }
        }
        let len = pts.last().map_or(1.0, |p| p.2);
        let speed = (len / PURSUIT_S).clamp(SPEED.0, SPEED.1);
        Path { pts, speed }
    }

    fn duration(&self) -> f64 {
        (self.pts.last().map_or(0.0, |p| p.2) / self.speed).min(PURSUIT_S + 2.0)
    }

    fn at(&self, t: f64) -> (f64, f64) {
        let s = (t.max(0.0) * self.speed).min(self.pts.last().map_or(0.0, |p| p.2));
        let i = self.pts.partition_point(|p| p.2 < s).clamp(1, self.pts.len() - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let k = if b.2 > a.2 { (s - a.2) / (b.2 - a.2) } else { 0.0 };
        (a.0 + k * (b.0 - a.0), a.1 + k * (b.1 - a.1))
    }
}

fn mm_per_px(mons: &[Monitor], x: f64, y: f64) -> f64 {
    let m = crate::monitors::at(mons, x, y);
    m.mm.map_or(0.25, |(w, _)| w / m.width as f64)
}

/// MediaPipe face mesh contours: face oval, subject's right eye, left eye (iris: 468-472 and 473-477, centre first).
const OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const RIGHT_EYE: [usize; 16] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const LEFT_EYE: [usize; 16] = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

/// The self view to draw: webcam texture (not mirrored), its aspect ratio, the face mesh (478 points, image fractions).
pub struct SelfView {
    pub tex: egui::TextureId,
    pub aspect: f32,
    pub face: Option<Vec<[f32; 2]>>,
}

/// Is the face well placed (as in WebGazer's face feedback box: inside a centred square of 66% of the image height)
/// and big enough (close enough for the eyes to be measurable)? A hint otherwise.
fn placement(face: Option<&Vec<[f32; 2]>>, aspect: f32) -> Result<(), &'static str> {
    let Some(f) = face else { return Err("no face detected: face the camera, add light") };
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    let outline: Vec<[f32; 2]> = if f.len() >= 478 { OVAL.iter().map(|&i| f[i]).collect() } else { f.clone() };
    for p in &outline {
        (x0, x1, y0, y1) = (x0.min(p[0]), x1.max(p[0]), y0.min(p[1]), y1.max(p[1]));
    }
    let half = 0.33 / aspect; // box half-width in image-width fractions
    if x0 < 0.5 - half || x1 > 0.5 + half || y0 < 0.17 || y1 > 0.83 {
        return Err("centre your face in the box");
    }
    if x1 - x0 < 0.15 {
        return Err("come a little closer");
    }
    Ok(())
}

/// One learned sample, with its raw landmarks so its features can be recomputed once the camera focal length is known.
struct Rec {
    f: [f64; NF],
    raw: Option<std::sync::Arc<Vec<f32>>>,
    target: [f64; 2],
    group: i64,
    weight: f64,
}

/// One validation dot: where, and the frames looked at it.
struct ValDot {
    target: (f64, f64),
    frames: Vec<[f64; NF]>,
}

pub struct Calibration {
    pub quick: bool,
    /// Quick calibration of screens just plugged in: fits their 3D pose.
    pub new_screens: bool,
    pub stage: Stage,
    pub points: Vec<(usize, f64, f64)>,
    pub model: GazeModel,
    pub umodel: GazeModel,
    /// Raw landmarks per dot, pointer paths, pursuits and validations, saved for offline evaluation (`--replay`).
    pub raw: Vec<serde_json::Value>,
    retried: std::collections::HashSet<(usize, i64, i64)>,
    mons: Vec<Monitor>,
    /// Pointer trail (t, x, y) and, per clicked dot, (dot index, shown at, clicked at).
    cursor: Vec<(f64, f64, f64)>,
    dots: Vec<(usize, f64, f64)>,
    /// Reading blocks (monitor, top-left x, y or bottom-left anchor, width, text index, anchored at the bottom) and,
    /// per block read, (block, shown at, clicked at).
    blocks: Vec<(usize, f64, f64, f64, usize, bool)>,
    reads: Vec<(usize, f64, f64)>,
    path: Option<Path>,
    pursuit_t0: f64,
    pursuits: usize,
    /// Frames (t, features) collected during the laser and pursuit phases: the tracker's history holds only ~10 s.
    buf: Vec<(f64, [f64; NF])>,
    /// Raw landmarks of the frames in `buf` (same order).
    raw_buf: Vec<Option<std::sync::Arc<Vec<f32>>>>,
    /// Every sample learned by the landmark model (dots, pointer paths, pursuits).
    recs: Vec<Rec>,
    ugs: Vec<crate::tracker::UgSample>,
    pulled: f64,
    /// Model so far (shown as the ring, used to align and filter moving samples).
    stage_model: Option<GazeModel>,
    /// During a pursuit: the model plus the pursuit frames so far, refitted every second; last frame time added.
    live: Option<GazeModel>,
    live_t: f64,
    ring: Option<(f64, f64)>,
    val_points: Vec<(f64, f64)>,
    val: Vec<ValDot>,
    /// Best model so far and its validation error: a pursuit that makes things worse is discarded.
    best: Option<(GazeModel, f64)>,
    pub report: Vec<String>,
    validation: Vec<String>,
}

impl Calibration {
    const QUICK_SETTLE: f64 = 0.5;
    const QUICK_COLLECT: f64 = 0.6;

    pub fn new(mons: &[Monitor]) -> Self {
        let mut points = Vec::new();
        for (mi, m) in mons.iter().enumerate() {
            // 3x3 per screen in a snake path
            points.extend((0..9).map(|k| {
                let (row, col) = (k / 3, if (k / 3) % 2 == 0 { k % 3 } else { 2 - k % 3 });
                // ~50-60 px from the edges: the gaze must learn to reach them
                let (fx, fy) = (0.025 + 0.475 * col as f64, 0.035 + 0.465 * row as f64);
                (mi, m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64)
            }));
        }
        let val_points = mons
            .iter()
            .flat_map(|m| VAL_SPOTS.iter().map(move |&(fx, fy)| (m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64)))
            .collect();
        let mut model = GazeModel::landmarks();
        model.layout = mons.iter().map(|m| m.layout()).collect();
        let mut umodel = GazeModel::unigaze();
        umodel.layout = model.layout.clone();
        Self {
            quick: false,
            new_screens: false,
            stage: Stage::Intro,
            points,
            model,
            umodel,
            raw: Vec::new(),
            retried: Default::default(),
            mons: mons.to_vec(),
            cursor: Vec::new(),
            dots: Vec::new(),
            blocks: Self::reading_blocks(mons),
            reads: Vec::new(),
            path: None,
            pursuit_t0: 0.0,
            pursuits: 0,
            buf: Vec::new(),
            raw_buf: Vec::new(),
            recs: Vec::new(),
            ugs: Vec::new(),
            pulled: 0.0,
            stage_model: None,
            live: None,
            live_t: 0.0,
            ring: None,
            val_points,
            val: Vec::new(),
            best: None,
            report: Vec::new(),
            validation: Vec::new(),
        }
    }

    /// Two reading blocks per screen, in opposite corners (alternating between screens) so the edges get covered.
    fn reading_blocks(mons: &[Monitor]) -> Vec<(usize, f64, f64, f64, usize, bool)> {
        let mut out = Vec::new();
        for (mi, m) in mons.iter().enumerate() {
            let w = (0.36 * m.width as f64).min(720.0);
            let (mx, my) = (0.025 * m.width as f64, 0.035 * m.height as f64);
            let corners = if mi % 2 == 0 { [(false, false), (true, true)] } else { [(true, false), (false, true)] };
            for (right, bottom) in corners {
                let x = if right { m.left as f64 + m.width as f64 - mx - w } else { m.left as f64 + mx };
                let y = if bottom { m.top as f64 + m.height as f64 - my } else { m.top as f64 + my };
                out.push((mi, x, y, w, (out.len() + mi) % TEXTS.len(), bottom));
            }
        }
        out
    }

    /// Pointer position at `t`, interpolated in the recorded trail.
    fn cursor_at(&self, t: f64) -> Option<(f64, f64)> {
        let k = self.cursor.partition_point(|c| c.0 < t);
        let (a, b) = (self.cursor.get(k.checked_sub(1)?)?, self.cursor.get(k)?);
        let u = (t - a.0) / (b.0 - a.0).max(1e-6);
        Some((a.1 + u * (b.1 - a.1), a.2 + u * (b.2 - a.2)))
    }

    /// Quick recalibration: dwell dots, centre and corners on a single screen, centre and two corners per screen
    /// otherwise.
    pub fn quick(mons: &[Monitor]) -> Self {
        let spots: &[(f64, f64)] = if mons.len() == 1 {
            &[(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        } else {
            &[(0.5, 0.5), (0.1, 0.15), (0.9, 0.85)]
        };
        let mut c = Self::new(mons);
        c.points = mons
            .iter()
            .enumerate()
            .flat_map(|(mi, m)| spots.iter().map(move |&(fx, fy)| (mi, m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64)))
            .collect();
        c.quick = true;
        c.stage = Stage::Quick { i: None, t0: 0.0, settle: 1.0 };
        c
    }

    /// Screens just plugged in (unknown to the gaze model): dwell dots on them only, centre and corners.
    pub fn for_screens(mons: &[Monitor], screens: &[Monitor]) -> Self {
        let mut c = Self::quick(mons);
        c.points = mons
            .iter()
            .enumerate()
            .filter(|(_, m)| screens.iter().any(|s| s.layout() == m.layout()))
            .flat_map(|(mi, m)| {
                [(0.5, 0.5), (0.08, 0.1), (0.92, 0.1), (0.92, 0.9), (0.08, 0.9)]
                    .into_iter()
                    .map(move |(fx, fy)| (mi, m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64))
            })
            .collect();
        c.new_screens = true;
        c
    }

    /// Something moves on screen (the laser, the pursuit dot, shrinking dots): redraw at the display rate.
    pub fn animating(&self) -> bool {
        matches!(self.stage, Stage::Laser { .. } | Stage::Reading { .. } | Stage::Pursuit { .. } | Stage::Validation { i: Some(_), .. } | Stage::Review | Stage::Quick { i: Some(_), .. })
    }

    pub fn hide_pointer(&self) -> bool {
        matches!(self.stage, Stage::Laser { .. } | Stage::Reading { .. })
    }

    pub fn reviewing(&self) -> bool {
        matches!(self.stage, Stage::Review)
    }

    pub fn on_pointer(&mut self, t: f64, x: f64, y: f64) {
        if matches!(self.stage, Stage::Laser { .. } | Stage::Reading { .. }) && self.cursor.last().is_none_or(|c| c.0 < t) {
            self.cursor.push((t, x, y));
        }
    }

    /// A click (global coordinates) in a calibration window.
    pub fn on_click(&mut self, t: f64, x: f64, y: f64, tracker: &Tracker) {
        match self.stage {
            Stage::Intro => self.stage = Stage::Laser { i: 0, since: t },
            Stage::PursuitIntro => self.start_pursuit(t),
            Stage::ReadingIntro => self.stage = Stage::Reading { i: 0, since: t },
            Stage::Reading { i, since } => {
                if t - since < 1.5 {
                    return; // a stray click right after the text appeared
                }
                self.reads.push((i, since, t));
                self.stage = if i + 1 < self.blocks.len() { Stage::Reading { i: i + 1, since: t } } else { Stage::Fitting { next: Next::Pursuit, drawn: false } };
            }
            Stage::Validation { i: None, .. } => self.stage = Stage::Validation { i: Some(0), t0: t },
            Stage::Quick { i: None, .. } => self.quick_dot(0, t),
            Stage::Laser { i, since } => {
                let (mi, dx, dy) = self.points[i];
                if (x - dx).hypot(y - dy) > HIT {
                    return;
                }
                let frames = tracker.samples_between(t - 0.35, t - 0.05);
                if frames.len() >= 3 {
                    for (_, f, raw) in frames {
                        self.learn(Rec { f, raw, target: [dx, dy], group: i as i64, weight: 1.0 });
                    }
                    let ug: Vec<[f64; 2]> = tracker.ug_between(t - 0.6, t - 0.02).iter().map(|u| u.angles).collect();
                    self.umodel.add(&ug, [dx, dy], i as i64, 1.0);
                    self.dots.push((i, since, t));
                    self.save_raw(tracker, mi, dx, dy, t - 0.35, t - 0.05);
                } else if self.retried.insert((mi, dx as i64, dy as i64)) {
                    self.points.push((mi, dx, dy)); // face not seen: this dot again at the end
                }
                self.stage = if i + 1 < self.points.len() {
                    Stage::Laser { i: i + 1, since: t }
                } else {
                    Stage::Fitting { next: Next::Reading, drawn: false }
                };
            }
            _ => {}
        }
    }

    /// R on the test screen: one more pursuit (then a new validation).
    pub fn redo_pursuit(&mut self) {
        if matches!(self.stage, Stage::Review) {
            self.stage = Stage::PursuitIntro;
        }
    }

    fn start_pursuit(&mut self, t: f64) {
        self.path = Some(Path::new(&self.mons));
        self.ring = None;
        self.pursuit_t0 = t;
        self.live = self.stage_model.as_ref().and_then(clone_model);
        self.live_t = t + 0.5;
        (self.buf, self.raw_buf, self.ugs, self.pulled) = (Vec::new(), Vec::new(), Vec::new(), t);
        self.stage = Stage::Pursuit { t0: t, next_refit: t + 1.5 };
    }

    fn quick_dot(&mut self, i: usize, t: f64) {
        let switched = i == 0 || self.points[i].0 != self.points[i - 1].0;
        self.stage = Stage::Quick { i: Some(i), t0: t, settle: if switched { 1.0 } else { Self::QUICK_SETTLE } };
    }

    fn save_raw(&mut self, tracker: &Tracker, mi: usize, x: f64, y: f64, t0: f64, t1: f64) {
        use base64::Engine;
        let frames = tracker.raw_between(t0, t1);
        // the features too: the eye patches can't be recomputed from the landmarks
        let feats: Vec<Vec<f64>> = tracker.samples_between(t0 - 1e-9, t1).iter().map(|(_, f, _)| f.to_vec()).collect();
        self.raw.push(serde_json::json!({
            "monitor": mi, "target": [x, y], "feats": feats,
            "frames": frames.iter().map(|f| f.0.as_slice().to_vec()).collect::<Vec<_>>(),
            "crops": frames.iter().map(|f| f.1.as_ref().map(|j| base64::engine::general_purpose::STANDARD.encode(j.as_slice()))).collect::<Vec<_>>(),
        }));
    }

    /// Keep the frames received since the last call (laser and pursuit phases). Frames are stamped when captured
    /// but reach the history ~20 ms later, after the face model: continue from the last frame received, not from
    /// the time of the last call, or frames still being processed would be skipped for good.
    fn pull(&mut self, t: f64, tracker: &Tracker) {
        if self.pulled == 0.0 {
            self.pulled = t;
        }
        let new = tracker.samples_between(self.pulled, f64::INFINITY);
        if let Some(&(last, _, _)) = new.last() {
            for (t, f, raw) in new {
                self.buf.push((t, f));
                self.raw_buf.push(raw);
            }
            self.ugs.extend(tracker.ug_between(self.pulled + 1e-9, last));
            self.pulled = last;
        }
    }

    fn learn(&mut self, r: Rec) {
        self.model.add(&[r.f], r.target, r.group, r.weight);
        self.recs.push(r);
    }

    /// Raw landmarks of the buffered frame captured at `t`.
    fn raw_at(&self, t: f64) -> Option<std::sync::Arc<Vec<f32>>> {
        let k = self.buf.partition_point(|(bt, _)| *bt < t);
        self.buf.get(k).filter(|(bt, _)| *bt == t).and_then(|_| self.raw_buf.get(k).cloned().flatten())
    }

    /// The geometric fit estimates the camera focal length (as the depth scale). Use it for the head pose itself:
    /// set it for the live features, recompute every sample's features from its raw landmarks, and refit.
    fn refocus(&mut self) {
        let Some(scale) = self.model.geo.as_ref().map(|g| g.depth_scale) else { return };
        let focal = crate::unigaze::focal() * scale;
        if (scale - 1.0).abs() < 0.05 {
            self.model.focal = Some(crate::unigaze::focal());
            return;
        }
        crate::unigaze::set_focal(focal);
        let mut model = GazeModel::landmarks();
        model.layout = self.model.layout.clone();
        let mut recomputed = 0;
        for r in &mut self.recs {
            if let Some(raw) = &r.raw {
                if raw.len() >= 478 * 3 + 2 {
                    let (w, h) = (raw[raw.len() - 2] as usize, raw[raw.len() - 1] as usize);
                    let mesh: Vec<[f32; 3]> = raw[..478 * 3].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                    // the eye patches don't depend on the focal length: kept
                    let mut f = crate::tracker::features_at(&mesh, w, h, focal);
                    f[crate::gaze::LAND..].copy_from_slice(&r.f[crate::gaze::LAND..]);
                    r.f = f;
                    recomputed += 1;
                }
            }
            model.add(&[r.f], r.target, r.group, r.weight);
        }
        model.focal = Some(focal);
        self.model = model;
        let err = self.model_with_geometry().map_or(f64::NAN, |(_, e)| e);
        self.report.push(format!(
            "Camera focal estimated at {focal:.0} px (at 1280 px width): head pose recomputed for {recomputed}/{} samples · calibration error {err:.0} px",
            self.recs.len()
        ));
        println!("{}", self.report.last().unwrap());
    }

    /// Smoothed live prediction of `model` for the latest frame.
    fn track(&mut self, tracker: &Tracker, live: bool) {
        let m = if live { self.live.as_ref() } else { self.stage_model.as_ref() };
        if let (Some(m), Some(f)) = (m, tracker.latest().and_then(|f| f.features)) {
            if let Some(p) = m.predict(&f) {
                let p = crate::monitors::clamp(&self.mons, p.0, p.1);
                self.ring = Some(match self.ring {
                    Some(r) => (r.0 + 0.35 * (p.0 - r.0), r.1 + 0.35 * (p.1 - r.1)),
                    None => p,
                });
            }
        }
    }

    pub fn update(&mut self, t: f64, tracker: &Tracker) -> Outcome {
        if matches!(self.stage, Stage::Laser { .. } | Stage::Reading { .. } | Stage::Pursuit { .. }) {
            self.pull(t, tracker);
        }
        match self.stage {
            Stage::Quick { i: Some(i), t0, settle } => {
                if t - t0 < settle + Self::QUICK_COLLECT {
                    return Outcome::Running;
                }
                let (mi, x, y) = self.points[i];
                let (a, b) = (t0 + settle, t0 + settle + Self::QUICK_COLLECT);
                let feats = tracker.features_between(a, b);
                if feats.len() >= 5 {
                    self.model.add(&feats, [x, y], i as i64, 1.0);
                    let ug: Vec<[f64; 2]> = tracker.ug_between(a, b).iter().map(|u| u.angles).collect();
                    self.umodel.add(&ug, [x, y], i as i64, 1.0);
                } else if self.retried.insert((mi, x as i64, y as i64)) {
                    self.points.push((mi, x, y));
                }
                if i + 1 >= self.points.len() {
                    return Outcome::Finished;
                }
                self.quick_dot(i + 1, t);
            }
            Stage::Fitting { next, drawn: false } => self.stage = Stage::Fitting { next, drawn: true },
            Stage::Fitting { next: Next::Reading, drawn: true } => {
                self.fit_phase1();
                self.stage = Stage::ReadingIntro;
            }
            Stage::Fitting { next: Next::Pursuit, drawn: true } => {
                self.fit_reading();
                self.stage = Stage::PursuitIntro;
            }
            Stage::Fitting { next: Next::Validation, drawn: true } => {
                self.finish_pursuit();
                self.val.clear();
                self.stage = Stage::Validation { i: None, t0: t };
            }
            Stage::Pursuit { t0, next_refit } => {
                self.track(tracker, true);
                if t >= next_refit {
                    self.live_refit(t - 0.4);
                    self.stage = Stage::Pursuit { t0, next_refit: next_refit + 1.0 };
                }
                if t - t0 >= self.path.as_ref().map_or(0.0, |p| p.duration()) {
                    self.stage = Stage::Fitting { next: Next::Validation, drawn: false };
                }
            }
            Stage::Validation { i: Some(i), t0 } => {
                let settle = if i == 0 || i % VAL_SPOTS.len() == 0 { 1.0 } else { VAL_SETTLE };
                if t - t0 >= settle + VAL_COLLECT {
                    let frames = tracker.features_between(t0 + settle, t0 + settle + VAL_COLLECT);
                    self.val.push(ValDot { target: self.val_points[i], frames });
                    if i + 1 < self.val_points.len() {
                        self.stage = Stage::Validation { i: Some(i + 1), t0: t };
                    } else {
                        self.judge();
                        self.ring = None;
                        self.stage = Stage::Review;
                    }
                }
            }
            Stage::Review => self.track(tracker, false),
            _ => {}
        }
        Outcome::Running
    }

    /// The pursuit frames so far, aligned with the target and filtered with `model`: (frame index, target).
    fn pursuit_samples(&self, model: &GazeModel, upto: f64) -> (f64, Vec<(f64, [f64; NF], (f64, f64))>) {
        let Some(path) = &self.path else { return (0.0, Vec::new()) };
        let t0 = self.pursuit_t0;
        let frames: Vec<(f64, [f64; NF])> = self.buf.iter().filter(|(t, _)| *t >= t0 + 0.5 && *t <= upto).copied().collect();
        let (lag, kept) = align(model, &frames, |t| (t >= t0).then(|| path.at(t - t0)));
        (lag, kept.into_iter().map(|(k, p)| (frames[k].0, frames[k].1, p)).collect())
    }

    fn pursuit_group(&self, t: f64) -> i64 {
        PURSUIT_GROUP + 50 * self.pursuits as i64 + (t - self.pursuit_t0) as i64
    }

    /// Every second of the pursuit: add the newly aligned frames to the live model and refit it (ridge only, the
    /// geometry is refitted at the end), so the ring converges while the user follows the dot.
    fn live_refit(&mut self, upto: f64) {
        let Some(live) = self.live.take() else { return };
        let (_, kept) = self.pursuit_samples(&live, upto);
        let mut live = live;
        let mut last = self.live_t;
        for (t, f, (x, y)) in kept.into_iter().filter(|s| s.0 > self.live_t) {
            live.add(&[f], [x, y], self.pursuit_group(t), 0.5);
            last = last.max(t);
        }
        self.live_t = last;
        let _ = live.refit();
        self.live = Some(live);
    }

    /// After the laser dots: fit on the clicks, then add the pointer paths where the gaze follows the pointer.
    fn fit_phase1(&mut self) {
        let Some((base, _)) = self.model_with_geometry() else {
            self.report.push("Not enough samples from the dots (is your face visible?)".into());
            return;
        };
        let mut trail = Vec::new();
        for &(i, since, clicked) in &self.dots {
            for &(t, f) in self.buf.iter().filter(|(t, _)| *t > since + 0.4 && *t < clicked - 0.35) {
                trail.push((t, f, i));
            }
        }
        let samples: Vec<(f64, [f64; NF])> = trail.iter().map(|(t, f, _)| (*t, *f)).collect();
        let (lag, kept) = align(&base, &samples, |t| self.cursor_at(t));
        for &(k, (x, y)) in &kept {
            let (t, f, i) = trail[k];
            let raw = self.raw_at(t);
            self.learn(Rec { f, raw, target: [x, y], group: MOVE_GROUP + i as i64, weight: 0.5 });
        }
        let err = self.model_with_geometry().map_or(f64::NAN, |(_, e)| e);
        self.report.push(format!(
            "Phase 1: {} dots clicked, {}/{} pointer-path frames kept (gaze {:.0} ms behind the pointer) · calibration error {err:.0} px",
            self.dots.len(),
            kept.len(),
            trail.len(),
            lag * 1000.0
        ));
        println!("{}", self.report.last().unwrap());
        self.refocus();
        // saved after the refocus: features consistent with the focal length recorded next to them
        let moves: Vec<Vec<f64>> = self
            .recs
            .iter()
            .filter(|r| (MOVE_GROUP..PURSUIT_GROUP).contains(&r.group))
            .map(|r| {
                let mut v = vec![0.0, r.target[0], r.target[1], (r.group - MOVE_GROUP) as f64];
                v.extend_from_slice(&r.f);
                v
            })
            .collect();
        self.raw.push(serde_json::json!({ "moves": moves, "lag": lag, "focal": crate::unigaze::focal() }));
    }

    /// After the reading: the frames where the gaze moves with the pointer under the words.
    fn fit_reading(&mut self) {
        let Some(base) = self.stage_model.as_ref().and_then(clone_model) else { return };
        let mut trail = Vec::new();
        for &(b, since, done) in &self.reads {
            for &(t, f) in self.buf.iter().filter(|(t, _)| *t > since + 0.6 && *t < done - 0.2) {
                trail.push((t, f, b));
            }
        }
        let samples: Vec<(f64, [f64; NF])> = trail.iter().map(|(t, f, _)| (*t, *f)).collect();
        let (lag, kept) = align_in(&base, &samples, |t| self.cursor_at(t), (-0.4, 0.4));
        let mut saved = Vec::new();
        for &(k, (x, y)) in &kept {
            let (t, f, b) = trail[k];
            let raw = self.raw_at(t);
            self.learn(Rec { f, raw, target: [x, y], group: READ_GROUP + b as i64, weight: 0.5 });
            let mut v = vec![t, x, y, (READ_GROUP - MOVE_GROUP) as f64 + b as f64];
            v.extend_from_slice(&f);
            saved.push(v);
        }
        self.raw.push(serde_json::json!({ "moves": saved, "lag": lag, "reading": true }));
        let err = self.model_with_geometry().map_or(f64::NAN, |(_, e)| e);
        self.report.push(format!(
            "Reading: {} texts, {}/{} frames with the eyes on the pointer (eye {:+.0} ms vs the pointer) · calibration error {err:.0} px",
            self.reads.len(),
            kept.len(),
            trail.len(),
            -lag * 1000.0
        ));
        println!("{}", self.report.last().unwrap());
    }

    /// End of a pursuit: align and filter the whole pursuit with the live model (the best estimate by now), learn
    /// it, and refit the geometry.
    fn finish_pursuit(&mut self) {
        let Some(live) = self.live.take() else { return };
        let (lag, kept) = self.pursuit_samples(&live, f64::INFINITY);
        let total = self.buf.iter().filter(|(t, _)| *t >= self.pursuit_t0 + 0.5).count();
        for &(t, f, (x, y)) in &kept {
            let (raw, group) = (self.raw_at(t), self.pursuit_group(t));
            self.learn(Rec { f, raw, target: [x, y], group, weight: 0.5 });
        }
        if let Some(path) = &self.path {
            for u in &self.ugs {
                if u.t >= self.pursuit_t0 + 0.5 + lag {
                    let (x, y) = path.at(u.t - lag - self.pursuit_t0);
                    self.umodel.add(&[u.angles], [x, y], PURSUIT_GROUP + 50 * self.pursuits as i64 + (u.t - self.pursuit_t0) as i64, 0.5);
                }
            }
            self.raw.push(serde_json::json!({
                "pursuit": self.pursuits + 1, "lag": lag,
                "path": (0..(path.duration() * 30.0) as usize).map(|i| { let p = path.at(i as f64 / 30.0); [p.0, p.1] }).collect::<Vec<_>>(),
                "frames": self.buf.iter().filter(|(t, _)| *t >= self.pursuit_t0).map(|(t, f)| { let mut v = vec![t - self.pursuit_t0]; v.extend_from_slice(f); v }).collect::<Vec<_>>(),
            }));
        }
        self.pursuits += 1;
        let err = self.model_with_geometry().map_or(f64::NAN, |(_, e)| e);
        self.report.push(format!(
            "Pursuit {}: {}/{total} frames with the eyes on the dot (eye {:.0} ms behind) · calibration error {err:.0} px",
            self.pursuits,
            kept.len(),
            lag * 1000.0
        ));
        println!("{}", self.report.last().unwrap());
    }

    /// Validation error of a model: median over the dots of each dot's median error (px).
    fn val_error(&self, m: &GazeModel) -> f64 {
        median(self.val.iter().filter_map(|d| self.dot_error(m, d)).collect())
    }

    fn dot_error(&self, m: &GazeModel, d: &ValDot) -> Option<f64> {
        let e: Vec<f64> = d
            .frames
            .iter()
            .filter_map(|f| m.predict(f))
            .map(|p| crate::monitors::clamp(&self.mons, p.0, p.1))
            .map(|p| (p.0 - d.target.0).hypot(p.1 - d.target.1))
            .collect();
        (!e.is_empty()).then(|| median(e))
    }

    /// After a validation: keep the current model if it beats the best one so far on it, else go back to the best.
    fn judge(&mut self) {
        let new = self.val_error(&self.model);
        let kept_new = match &self.best {
            Some((b, _)) => {
                let old = self.val_error(b);
                if new <= old {
                    true
                } else {
                    self.report.push(format!("That pursuit made it worse on this validation ({new:.0} px vs {old:.0} px): kept the previous model"));
                    println!("{}", self.report.last().unwrap());
                    if let Some(m) = clone_model(b) {
                        self.model = m;
                    }
                    false
                }
            }
            None => true,
        };
        if kept_new {
            self.best = clone_model(&self.model).map(|m| (m, new));
        }
        self.stage_model = clone_model(&self.model);
        self.validation = self.describe_validation();
        self.raw.push(serde_json::json!({
            "validation": self.val.iter().map(|d| serde_json::json!({ "target": [d.target.0, d.target.1], "frames": d.frames.iter().map(|f| f.to_vec()).collect::<Vec<_>>() })).collect::<Vec<_>>(),
        }));
        for line in &self.validation {
            println!("validation: {line}");
        }
    }

    /// Per screen: error in the centre and in the corners, in px and degrees.
    fn describe_validation(&self) -> Vec<String> {
        let m = &self.model;
        let dist = {
            let z: Vec<f64> = self.val.iter().flat_map(|d| d.frames.iter().map(|f| f[19])).collect();
            let scale = m.geo.as_ref().map_or(1.0, |g| g.depth_scale);
            median(z) * scale
        };
        let deg = |px: f64, x: f64, y: f64| (px * mm_per_px(&self.mons, x, y) / dist.max(200.0)).atan().to_degrees();
        let mut out = vec![format!("validation error {:.0} px (median of {} dots)", self.val_error(m), self.val.len())];
        for (mi, mon) in self.mons.iter().enumerate() {
            let dots: Vec<&ValDot> = self.val.iter().filter(|d| crate::monitors::at(&self.mons, d.target.0, d.target.1).layout() == mon.layout()).collect();
            let errs = |corner: bool| -> Vec<(f64, f64, f64)> {
                dots.iter()
                    .filter(|d| {
                        let cx = mon.left as f64 + mon.width as f64 / 2.0;
                        ((d.target.0 - cx).abs() > mon.width as f64 / 4.0) == corner
                    })
                    .filter_map(|d| Some((self.dot_error(m, d)?, d.target.0, d.target.1)))
                    .collect()
            };
            let fmt = |v: Vec<(f64, f64, f64)>| {
                if v.is_empty() {
                    return "-".to_string();
                }
                let px = median(v.iter().map(|e| e.0).collect());
                format!("{px:.0} px ({:.1}°)", deg(px, v[0].1, v[0].2))
            };
            out.push(format!("screen {} ({}x{}): centre {} · corners {}", mi + 1, mon.width, mon.height, fmt(errs(false)), fmt(errs(true))));
        }
        out
    }

    /// Fit the geometry and the model on everything so far; keeps it as the stage model. Returns a copy and its
    /// cross-validated error (px).
    fn model_with_geometry(&mut self) -> Option<(GazeModel, f64)> {
        match self.model.fit_geometry(&self.mons).and_then(|(_, what)| self.model.fit().map(|e| (e, what))) {
            Ok((err, what)) => {
                println!("geometry: {what} · cross-validated error {err:.0} px");
                self.stage_model = clone_model(&self.model);
                clone_model(&self.model).map(|m| (m, err))
            }
            Err(e) => {
                println!("calibration fit failed: {e:#}");
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, p: &Painter, mi: usize, m: &Monitor, size: egui::Vec2, face: bool, t: f64, pointer: Option<(f64, f64)>, view: Option<&SelfView>) {
        let local = |x: f64, y: f64| Pos2::new((x - m.left as f64) as f32, (y - m.top as f64) as f32);
        let here = |x: f64, y: f64| crate::monitors::at(&self.mons, x, y).layout() == m.layout();
        let centre = Pos2::new(size.x / 2.0, size.y / 2.0);
        let text = |s: &str| {
            p.text(centre, Align2::CENTER_CENTER, s, FontId::proportional(24.0), Color32::WHITE);
        };
        let top = |s: &str| {
            p.text(Pos2::new(size.x / 2.0, 30.0), Align2::CENTER_CENTER, s, FontId::proportional(15.0), Color32::GRAY);
        };
        let ring = |p: &Painter| {
            if let Some((gx, gy)) = self.ring.filter(|r| here(r.0, r.1)) {
                p.circle_stroke(local(gx, gy), 40.0, Stroke::new(3.0, RED));
            }
        };
        let report = || self.report.iter().map(|r| format!("{r}\n")).collect::<String>();
        if !face && !matches!(self.stage, Stage::Intro | Stage::Review | Stage::Fitting { .. }) {
            p.text(Pos2::new(size.x / 2.0, 60.0), Align2::CENTER_CENTER, "Face not detected: face the camera, add light", FontId::proportional(18.0), Color32::from_rgb(255, 180, 0));
        }
        // self view on the camera's screen, top right of centre: clear of the dots (2.5% / 50% / 97.5%), of the reading
        // blocks (top-left and bottom-right corners of this screen) and of the centred instructions
        let camera_screen = crate::monitors::camera_first(&self.mons)[0].layout() == m.layout();
        if let (Some(v), true, false) = (view, camera_screen, matches!(self.stage, Stage::Fitting { .. })) {
            let w = 320.0f32.min(size.x * 0.2);
            let r = egui::Rect::from_min_size(Pos2::new(size.x * 0.56, size.y * 0.1), egui::vec2(w, w / v.aspect));
            // mirrored, like a mirror: moving left moves the image left
            p.image(v.tex, r, egui::Rect::from_min_max(Pos2::new(1.0, 0.0), Pos2::new(0.0, 1.0)), Color32::WHITE);
            let at = |q: [f32; 2]| Pos2::new(r.min.x + (1.0 - q[0]) * r.width(), r.min.y + q[1] * r.height());
            if let Some(f) = v.face.as_ref().filter(|f| f.len() >= 478) {
                let green = Color32::from_rgb(126, 231, 135);
                for &q in &f[..468] {
                    p.circle_filled(at(q), 0.7, Color32::from_rgba_unmultiplied(126, 231, 135, 90));
                }
                let contour = |idx: &[usize]| egui::Shape::closed_line(idx.iter().map(|&i| at(f[i])).collect(), Stroke::new(1.2, green));
                p.add(contour(&OVAL));
                p.add(contour(&RIGHT_EYE));
                p.add(contour(&LEFT_EYE));
                for c in [468, 473] {
                    // iris: centre and radius from its four edge points
                    let centre = at(f[c]);
                    let radius = (1..=4).map(|k| at(f[c + k]).distance(centre)).sum::<f32>() / 4.0;
                    p.circle_stroke(centre, radius.max(1.5), Stroke::new(1.2, Color32::from_rgb(90, 200, 255)));
                    p.circle_filled(centre, 1.2, Color32::from_rgb(90, 200, 255));
                }
            }
            let ok = placement(v.face.as_ref(), v.aspect);
            let color = if ok.is_ok() { Color32::from_rgb(126, 231, 135) } else { RED };
            let side = r.height() * 0.66;
            p.rect_stroke(egui::Rect::from_center_size(r.center(), egui::vec2(side, side)), 2.0, Stroke::new(2.0, color), egui::StrokeKind::Middle);
            p.text(Pos2::new(r.center().x, r.max.y + 14.0), Align2::CENTER_CENTER, ok.err().unwrap_or("face well placed"), FontId::proportional(14.0), color);
        }
        match self.stage {
            Stage::Intro => text(&format!(
                "Calibration · {} screen(s) · step 1/4\n\nYour mouse becomes a red LASER. For each dot: move the laser onto it\nwhile following the laser with your EYES, then CLICK the dot.\n\nYou don't have to sit still: lean and shift on your chair between dots,\nit teaches precog where you are.\n\nCLICK anywhere to start · ESC to cancel",
                self.mons.len()
            )),
            Stage::Laser { i, since } => {
                let (pm, x, y) = self.points[i];
                if pm == mi {
                    let c = local(x, y);
                    let pulse = (1.0 + ((t - since) * 4.0).sin() as f32) * 3.0;
                    p.circle_stroke(c, 16.0 + pulse, Stroke::new(2.0, Color32::from_gray(160)));
                    p.circle_filled(c, 7.0, Color32::WHITE);
                }
                top(&format!("Step 1/4 · dot {}/{} · bring the laser onto the dot, eyes on the laser, then click it", i + 1, self.points.len()));
                if let Some((x, y)) = pointer.filter(|q| here(q.0, q.1)) {
                    let c = local(x, y);
                    p.circle_filled(c, 14.0, Color32::from_rgba_unmultiplied(255, 59, 48, 60));
                    p.circle_filled(c, 6.0, RED);
                }
            }
            Stage::Fitting { .. } => text("Fitting the model…"),
            Stage::ReadingIntro => text(&format!(
                "{}\nStep 2/4 · reading\n\nShort texts will appear in different places. Read each one normally, moving the red laser\n(your mouse) under the words as you read them, then CLICK when you reach the end.\n\nCLICK anywhere to start · ESC to cancel",
                report()
            )),
            Stage::Reading { i, .. } => {
                let (bm, x, y, w, k, bottom) = self.blocks[i];
                if bm == mi {
                    let galley = p.layout(TEXTS[k].to_string(), FontId::proportional(22.0), Color32::WHITE, w as f32);
                    let h = galley.size().y as f64;
                    let top_y = if bottom { y - h } else { y };
                    p.galley(local(x, top_y), galley, Color32::WHITE);
                }
                top(&format!("Step 2/4 · text {}/{} · read it with the laser under the words, then click", i + 1, self.blocks.len()));
                if let Some((x, y)) = pointer.filter(|q| here(q.0, q.1)) {
                    let c = local(x, y);
                    p.circle_filled(c, 12.0, Color32::from_rgba_unmultiplied(255, 59, 48, 60));
                    p.circle_filled(c, 5.0, RED);
                }
            }
            Stage::PursuitIntro => text(&format!(
                "{}\nStep 3/4 · pursuit\n\nFollow the moving dot with your EYES for ~{PURSUIT_S:.0} s. Your head may move.\nThe red ring shows where precog thinks you look; it learns as you go and should\nget closer and closer. Keep your eyes on the DOT, not on the ring.\n\nCLICK anywhere to start · ESC to cancel",
                report()
            )),
            Stage::Pursuit { t0, .. } => {
                if let Some(path) = &self.path {
                    let (x, y) = path.at(t - t0);
                    if here(x, y) {
                        let c = local(x, y);
                        p.circle_filled(c, 9.0, Color32::WHITE);
                        p.circle_stroke(c, 13.0, Stroke::new(2.0, Color32::from_gray(140)));
                    }
                    let frac = ((t - t0) / path.duration()).clamp(0.0, 1.0) as f32;
                    p.rect_filled(egui::Rect::from_min_size(Pos2::new(0.0, size.y - 4.0), egui::vec2(size.x * frac, 4.0)), 0.0, Color32::from_gray(90));
                    ring(p);
                    top("Step 3/4 · follow the white dot with your eyes (the red ring is precog's guess, learning live)");
                }
            }
            Stage::Validation { i: None, .. } => text(&format!(
                "{}\nStep 4/4 · validation\n\nLook at each dot until the next one appears (~1.5 s each).\nThis measures the accuracy; nothing is learned from it.\n\nCLICK anywhere to start · ESC to cancel",
                report()
            )),
            Stage::Validation { i: Some(i), t0 } => {
                let (x, y) = self.val_points[i];
                if here(x, y) {
                    let el = t - t0;
                    let r = (30.0 * (1.0 - el / 0.6)).max(6.0) as f32;
                    p.circle_filled(local(x, y), r, Color32::from_rgb(90, 200, 255));
                    p.circle_filled(local(x, y), 2.0, Color32::WHITE);
                }
                top(&format!("Step 4/4 · validation {}/{} · eyes on the dot", i + 1, self.val_points.len()));
            }
            Stage::Review => {
                for &(x, y) in self.val_points.iter().filter(|q| here(q.0, q.1)) {
                    p.circle_stroke(local(x, y), 10.0, Stroke::new(2.0, Color32::from_gray(200)));
                    p.circle_filled(local(x, y), 3.0, Color32::from_gray(200));
                }
                ring(p);
                if mi == 0 {
                    p.text(
                        Pos2::new(size.x / 2.0, size.y * 0.3),
                        Align2::CENTER_CENTER,
                        format!(
                            "Test · look at the markers on every screen: does the red ring follow your eyes and reach the corners?\n\n{}\n{}\nY: yes, done · R: one more pursuit (kept only if it does better) · A: start over · ESC: cancel",
                            report(),
                            self.validation.iter().map(|r| format!("{r}\n")).collect::<String>()
                        ),
                        FontId::proportional(16.0),
                        Color32::WHITE,
                    );
                }
            }
            Stage::Quick { i: None, .. } => text(&if self.new_screens {
                format!(
                    "New screen detected · {} dots\n\nprecog needs to learn where this screen is. Follow each dot with your EYES.\n(Your calibration on the other screens is kept.)\n\nCLICK anywhere to start · ESC to skip",
                    self.points.len()
                )
            } else {
                format!("Quick recalibration · {} dots\n\nStay as you are and follow each dot with your EYES.\n\nCLICK anywhere to start · ESC to cancel", self.points.len())
            }),
            Stage::Quick { i: Some(i), t0, settle } => {
                let (pm, x, y) = self.points[i];
                if pm == mi {
                    let el = t - t0;
                    let r = if el < settle { (34.0 * (1.0 - el / settle)).max(6.0) } else { 6.0 } as f32;
                    p.circle_filled(local(x, y), r, RED);
                    p.circle_filled(local(x, y), 2.0, Color32::WHITE);
                }
                top(&format!("{}/{} · eyes on the dot", i + 1, self.points.len()));
            }
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn clone_model(m: &GazeModel) -> Option<GazeModel> {
    let mut c = serde_json::from_value::<GazeModel>(serde_json::to_value(m).ok()?).ok()?;
    c.refit().ok()?;
    Some(c)
}

/// Frames where the gaze moves with the target (Pfeuffer et al.'s pursuit test): in a +-0.3 s window, the predicted
/// gaze correlates with the target on each axis the target moves along (> 40 px). Correlation ignores a constant
/// offset or a scale error, so this works with an imperfect model; a glance away, a blink or a saccade breaks it.
/// Plus a loose distance check (gross outliers).
fn attending(times: &[f64], preds: &[(f64, f64)], targets: &[(f64, f64)]) -> Vec<bool> {
    let n = preds.len();
    let dist: Vec<f64> = (0..n).map(|k| (preds[k].0 - targets[k].0).hypot(preds[k].1 - targets[k].1)).collect();
    let far = (3.0 * median(dist.clone())).max(250.0);
    (0..n)
        .map(|k| {
            if dist[k] > far {
                return false;
            }
            let idx: Vec<usize> = (0..n).filter(|&j| (times[j] - times[k]).abs() <= 0.3).collect();
            if idx.len() < 8 {
                return false;
            }
            let mut tested = false;
            for axis in 0..2 {
                let a = |j: usize, v: &[(f64, f64)]| if axis == 0 { v[j].0 } else { v[j].1 };
                let (lo, hi) = idx.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &j| (lo.min(a(j, targets)), hi.max(a(j, targets))));
                if hi - lo < 40.0 {
                    continue;
                }
                tested = true;
                let (mp, mt) = (idx.iter().map(|&j| a(j, preds)).sum::<f64>() / idx.len() as f64, idx.iter().map(|&j| a(j, targets)).sum::<f64>() / idx.len() as f64);
                let (mut cov, mut vp, mut vt) = (0.0, 0.0, 0.0);
                for &j in &idx {
                    let (dp, dt) = (a(j, preds) - mp, a(j, targets) - mt);
                    cov += dp * dt;
                    vp += dp * dp;
                    vt += dt * dt;
                }
                if cov / (vp * vt).sqrt().max(1e-9) < 0.5 {
                    return false;
                }
            }
            tested
        })
        .collect()
}

/// Align moving-target frames with the target: the lag (0-500 ms) that best fits the model's predicted gaze to the
/// target, then the frames where the gaze is on the target (`attending`). Returns the lag and (frame index, target
/// position at t - lag).
fn align(model: &GazeModel, frames: &[(f64, [f64; NF])], target: impl Fn(f64) -> Option<(f64, f64)>) -> (f64, Vec<(usize, (f64, f64))>) {
    align_in(model, frames, target, (0.0, 0.5))
}

/// `align` with a lag range (s): negative = the gaze runs ahead of the target (reading ahead of the pointer).
fn align_in(model: &GazeModel, frames: &[(f64, [f64; NF])], target: impl Fn(f64) -> Option<(f64, f64)>, lags: (f64, f64)) -> (f64, Vec<(usize, (f64, f64))>) {
    let preds: Vec<Option<(f64, f64)>> = frames.iter().map(|(_, f)| model.predict(f)).collect();
    let pairs = |lag: f64| -> Vec<(usize, (f64, f64), (f64, f64))> {
        frames.iter().enumerate().filter_map(|(k, (t, _))| Some((k, preds[k]?, target(t - lag)?))).collect()
    };
    let steps = ((lags.1 - lags.0) * 60.0).round() as i32;
    let lag = (0..=steps)
        .map(|i| lags.0 + i as f64 / 60.0)
        .min_by(|a, b| {
            let e = |lag: f64| median(pairs(lag).iter().map(|(_, p, q)| (p.0 - q.0).hypot(p.1 - q.1)).collect());
            e(*a).total_cmp(&e(*b))
        })
        .unwrap_or(0.1);
    let v = pairs(lag);
    let times: Vec<f64> = v.iter().map(|(k, _, _)| frames[*k].0).collect();
    let keep = attending(&times, &v.iter().map(|x| x.1).collect::<Vec<_>>(), &v.iter().map(|x| x.2).collect::<Vec<_>>());
    (lag, v.into_iter().zip(keep).filter(|(_, k)| *k).map(|((k, _, q), _)| (k, q)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pursuit_path_is_smooth_and_on_screen() {
        let mons = vec![
            Monitor { left: 0, top: 1440, width: 1920, height: 1080, primary: true, mm: Some((294.0, 165.0)), builtin: true },
            Monitor { left: 0, top: 0, width: 2560, height: 1440, primary: false, mm: Some((597.0, 336.0)), builtin: false },
        ];
        let p = Path::new(&mons);
        let d = p.duration();
        assert!((10.0..=PURSUIT_S + 2.0).contains(&d), "duration {d:.1}s");
        let single = Path::new(&mons[..1]).duration();
        assert!(single >= 12.0, "a single screen still gets a long enough pursuit ({single:.1}s)");
        let mut prev = p.at(0.0);
        for i in 1..(d * 60.0) as usize {
            let q = p.at(i as f64 / 60.0);
            let step_mm = (q.0 - prev.0).hypot(q.1 - prev.1) * mm_per_px(&mons, q.0, q.1);
            assert!(step_mm < SPEED.1 / 60.0 * 1.6, "jump of {step_mm:.1} mm at {i}");
            assert!(mons.iter().any(|m| m.contains(q.0, q.1)) || (q.1 - 1440.0).abs() < 60.0, "off screen at {q:?}");
            prev = q;
        }
    }
}

    #[test]
    fn attention_ignores_offset_but_not_glances() {
        // gaze follows the target with a constant 200 px offset, except a glance away in the middle
        let times: Vec<f64> = (0..90).map(|i| i as f64 / 30.0).collect();
        let targets: Vec<(f64, f64)> = times.iter().map(|t| (500.0 + 300.0 * t, 400.0 + 100.0 * (2.0 * t).sin())).collect();
        let mut preds: Vec<(f64, f64)> = targets.iter().map(|q| (q.0 + 200.0, q.1 - 50.0)).collect();
        for p in preds.iter_mut().skip(40).take(12) {
            *p = (1800.0, 900.0);
        }
        let keep = attending(&times, &preds, &targets);
        assert!(keep[..30].iter().filter(|&&k| k).count() > 20, "offset alone must not reject");
        assert!(keep[42..50].iter().all(|&k| !k), "the glance must be rejected");
    }

#[test]
fn reading_blocks_cover_opposite_corners_inside_the_screens() {
    let mons = vec![
        Monitor { left: 0, top: 1440, width: 1920, height: 1080, primary: true, mm: Some((294.0, 165.0)), builtin: true },
        Monitor { left: 0, top: 0, width: 2560, height: 1440, primary: false, mm: Some((597.0, 336.0)), builtin: false },
    ];
    let b = Calibration::reading_blocks(&mons);
    assert_eq!(b.len(), 4);
    for &(mi, x, y, w, k, _) in &b {
        let m = mons[mi];
        assert!(x >= m.left as f64 && x + w <= m.right() as f64 && y >= m.top as f64 && y <= m.bottom() as f64);
        assert!(k < TEXTS.len());
    }
    // per screen: one block on the left and one on the right, one at the top and one at the bottom
    for mi in 0..2 {
        let s: Vec<_> = b.iter().filter(|q| q.0 == mi).collect();
        assert!(s[0].1 != s[1].1 && s[0].5 != s[1].5);
    }
}

#[test]
fn face_placement_hints() {
    // a 478-point mesh whose oval spans a square of side w around (cx, cy)
    let face = |cx: f32, cy: f32, w: f32| {
        let mut f = vec![[cx, cy]; 478];
        for (k, &i) in OVAL.iter().enumerate() {
            let a = k as f32 / OVAL.len() as f32 * std::f32::consts::TAU;
            f[i] = [cx + w / 2.0 * a.cos(), cy + w / 2.0 * a.sin()];
        }
        Some(f)
    };
    let aspect = 16.0 / 9.0;
    assert!(placement(face(0.5, 0.5, 0.2).as_ref(), aspect).is_ok());
    assert_eq!(placement(face(0.15, 0.5, 0.2).as_ref(), aspect), Err("centre your face in the box"));
    assert_eq!(placement(face(0.5, 0.5, 0.1).as_ref(), aspect), Err("come a little closer"));
    assert!(placement(None, aspect).is_err());
}
