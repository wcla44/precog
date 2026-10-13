//! Camera thread: face mesh -> gaze feature vector, hand landmarks -> pose + swipes, plus mouse-click polling
//! (clicks are the implicit calibration signal).

use crate::gaze::NF;
use crate::gestures::{classify, Pose, Swipe, SwipeDetector};
use crate::now;
use crate::vision::{FaceMesh, HandTracker, IrisModel, Rgb};
use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct Frame {
    pub t: f64,
    pub features: Option<[f64; NF]>,
    pub pose: Option<Pose>,
    /// Raw face landmarks (478 x [x, y, z], iris centres refined) + [w, h]: saved with calibrations for offline tuning.
    pub raw: Option<Arc<Vec<f32>>>,
}

#[derive(Default)]
pub struct Shared {
    pub latest: Option<Frame>,
    pub history: VecDeque<Frame>,
    pub fps: f32,
    pub error: Option<String>,
}

pub struct Tracker {
    pub shared: Arc<Mutex<Shared>>,
    pub swipes: Receiver<(f64, Swipe)>,
    /// (t, x, y) of left-button presses, global logical coordinates.
    pub clicks: Receiver<(f64, f64, f64)>,
    pub use_hands: Arc<AtomicBool>,
    pub mirrored: Arc<AtomicBool>,
}

// subject's right eye, then left eye: corners, upper/lower lid, iris centre
const EYES: [(usize, usize, usize, usize, usize); 2] = [(33, 133, 159, 145, 468), (362, 263, 386, 374, 473)];

fn eye_features(p: &[[f32; 3]], (c1, c2, top, bot, iris): (usize, usize, usize, usize, usize)) -> [f64; 3] {
    let (a, b) = (p[c1], p[c2]);
    let (ax, ay) = (b[0] - a[0], b[1] - a[1]);
    let n = (ax * ax + ay * ay).sqrt().max(1e-6);
    let (ux, uy) = (ax / n, ay / n);
    let (dx, dy) = (p[iris][0] - (a[0] + b[0]) / 2.0, p[iris][1] - (a[1] + b[1]) / 2.0);
    let open = ((p[top][0] - p[bot][0]).powi(2) + (p[top][1] - p[bot][1]).powi(2)).sqrt() / n;
    [((dx * ux + dy * uy) / n) as f64, ((-dx * uy + dy * ux) / n) as f64, open as f64]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Head yaw / pitch from the face's own 3D axes (eye line, forehead->chin line), insensitive to 2D foreshortening.
fn head_angles_3d(p: &[[f32; 3]]) -> (f32, f32) {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let x = norm3(sub(p[263], p[33]));
    let down = sub(p[152], p[10]);
    let d = down[0] * x[0] + down[1] * x[1] + down[2] * x[2];
    let y = norm3([down[0] - d * x[0], down[1] - d * x[1], down[2] - d * x[2]]);
    (y[2].clamp(-1.0, 1.0).asin(), x[2].clamp(-1.0, 1.0).asin()) // (pitch, yaw)
}

/// 14 features: iris offsets + eyelid openness for both eyes, head pose (3D axes + 2D geometry), face position/scale.
pub fn features(p: &[[f32; 3]], w: usize, h: usize) -> [f64; NF] {
    let [rx, ry, ro] = eye_features(p, EYES[0]);
    let [lx, ly, lo] = eye_features(p, EYES[1]);
    let (e1, e2) = (p[33], p[263]);
    let roll = (e2[1] - e1[1]).atan2(e2[0] - e1[0]);
    let (c, s) = (roll.cos(), roll.sin());
    let rot = |q: [f32; 3]| (q[0] * c + q[1] * s, -q[0] * s + q[1] * c);
    let (nose, chin, forehead, cl, cr) = (rot(p[1]), rot(p[152]), rot(p[10]), rot(p[234]), rot(p[454]));
    let face_w = (cr.0 - cl.0).abs().max(1e-6);
    let face_h = (chin.1 - forehead.1).abs().max(1e-6);
    let yaw = (nose.0 - (cl.0 + cr.0) / 2.0) / face_w;
    let pitch = (nose.1 - (forehead.1 + chin.1) / 2.0) / face_h;
    let (pitch3, yaw3) = head_angles_3d(p);
    let cx = (p[234][0] + p[454][0]) / 2.0 / w as f32 - 0.5;
    let cy = (p[10][1] + p[152][1]) / 2.0 / h as f32 - 0.5;
    [
        rx, ry, ro, lx, ly, lo,
        pitch3 as f64, yaw3 as f64, roll as f64,
        cx as f64, cy as f64, (face_w / w as f32) as f64,
        yaw as f64, pitch as f64,
    ]
}

impl Tracker {
    pub fn start(camera: u32, models: std::path::PathBuf) -> Tracker {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (swipe_tx, swipes) = unbounded();
        let (click_tx, clicks) = unbounded();
        let use_hands = Arc::new(AtomicBool::new(true));
        let mirrored = Arc::new(AtomicBool::new(false));
        let t = Tracker { shared: shared.clone(), swipes, clicks, use_hands: use_hands.clone(), mirrored: mirrored.clone() };
        std::thread::Builder::new()
            .name("tracker".into())
            .spawn(move || {
                if let Err(e) = run(camera, &models, &shared, swipe_tx, use_hands, mirrored) {
                    shared.lock().error = Some(format!("{e:#}"));
                }
            })
            .unwrap();
        std::thread::Builder::new().name("clicks".into()).spawn(move || poll_clicks(click_tx)).unwrap();
        t
    }

    pub fn latest(&self) -> Option<Frame> {
        self.shared.lock().latest.clone()
    }

    pub fn raw_between(&self, t0: f64, t1: f64) -> Vec<Arc<Vec<f32>>> {
        self.shared.lock().history.iter().filter(|f| f.t >= t0 && f.t <= t1).filter_map(|f| f.raw.clone()).collect()
    }

    pub fn features_between(&self, t0: f64, t1: f64) -> Vec<[f64; NF]> {
        self.shared.lock().history.iter().filter(|f| f.t >= t0 && f.t <= t1).filter_map(|f| f.features).collect()
    }
}

fn run(
    camera: u32,
    models: &std::path::Path,
    shared: &Mutex<Shared>,
    swipe_tx: Sender<(f64, Swipe)>,
    use_hands: Arc<AtomicBool>,
    mirrored: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut face = FaceMesh::new(models)?;
    let mut iris = IrisModel::new(models).map_err(|e| eprintln!("iris model unavailable ({e:#}): using face-mesh iris")).ok();
    let mut hands: Option<HandTracker> = None;
    let cam = Cam::open(camera).map_err(|e| {
        anyhow::anyhow!("{e:#}. Grant Camera access to your terminal app (macOS: System Settings > Privacy & Security > Camera)")
    })?;
    let mut swipe = SwipeDetector::new();
    let (mut n, mut t_fps, mut frames) = (0u64, now(), 0u32);
    let mut pose = None;
    loop {
        let img = cam.frame()?;
        let t = now();
        let (w, h) = (img.width, img.height);
        let rgb = Rgb { w, h, data: &img.data };
        let mut mesh = face.process(&rgb)?;
        if let (Some(p), Some(iris)) = (mesh.as_mut(), iris.as_mut()) {
            let [r, l] = iris.process(&rgb, p)?;
            p[468][..2].copy_from_slice(&r);
            p[473][..2].copy_from_slice(&l);
        }
        let feats = mesh.as_ref().map(|p| features(p, w, h));
        let raw = mesh.map(|p| {
            let mut v: Vec<f32> = p.iter().flatten().copied().collect();
            v.extend([w as f32, h as f32]);
            Arc::new(v)
        });

        if use_hands.load(Ordering::Relaxed) {
            if hands.is_none() {
                hands = Some(HandTracker::new(models)?);
            }
            if n % 2 == 0 {
                // hands every other frame: saves CPU
                let lm = hands.as_mut().unwrap().process(&rgb)?;
                let pts: Option<Vec<[f32; 2]>> = lm.map(|p| p.iter().map(|q| [q[0] / h as f32, q[1] / h as f32]).collect());
                pose = pts.as_ref().map(|p| classify(p));
                swipe.mirrored = mirrored.load(Ordering::Relaxed);
                if let Some(dir) = swipe.update(t, pts.as_ref().map(|p| p[0])) {
                    let _ = swipe_tx.send((t, dir));
                }
            }
        } else {
            pose = None;
        }

        let f = Frame { t, features: feats, pose, raw };
        {
            let mut s = shared.lock();
            s.history.push_back(f.clone());
            while s.history.len() > 300 {
                s.history.pop_front();
            }
            s.latest = Some(f);
        }
        n += 1;
        frames += 1;
        if t - t_fps >= 1.0 {
            shared.lock().fps = frames as f32 / (t - t_fps) as f32;
            frames = 0;
            t_fps = t;
        }
    }
}

#[cfg(target_os = "macos")]
use crate::camera_mac::Camera as Cam;

/// Linux: V4L2 through nokhwa, same interface as the macOS capture.
#[cfg(not(target_os = "macos"))]
struct Cam(std::sync::Mutex<nokhwa::Camera>);

#[cfg(not(target_os = "macos"))]
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[cfg(not(target_os = "macos"))]
impl Cam {
    fn open(index: u32) -> anyhow::Result<Self> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
        let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut cam = nokhwa::Camera::new(CameraIndex::Index(index), fmt)?;
        cam.open_stream()?;
        Ok(Self(std::sync::Mutex::new(cam)))
    }

    fn frame(&self) -> anyhow::Result<RgbFrame> {
        use nokhwa::pixel_format::RgbFormat;
        let img = self.0.lock().unwrap().frame()?.decode_image::<RgbFormat>()?;
        Ok(RgbFrame { width: img.width() as usize, height: img.height() as usize, data: img.into_raw() })
    }
}

/// Mouse-button polling (no global event tap needed). Needs Accessibility on macOS; silently idle without it.
fn poll_clicks(tx: Sender<(f64, f64, f64)>) {
    use device_query::DeviceQuery;
    #[cfg(target_os = "macos")]
    let state = loop {
        if let Some(s) = device_query::DeviceState::checked_new() {
            break s;
        }
        std::thread::sleep(std::time::Duration::from_secs(5)); // permission may be granted later
    };
    #[cfg(not(target_os = "macos"))]
    let state = device_query::DeviceState::new();
    let mut was_down = false;
    loop {
        let m = state.get_mouse();
        let down = m.button_pressed.get(1).copied().unwrap_or(false);
        if down && !was_down {
            let _ = tx.send((now(), m.coords.0 as f64, m.coords.1 as f64));
        }
        was_down = down;
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
}
