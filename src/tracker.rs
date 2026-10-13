//! Camera thread: face mesh -> gaze feature vector, hand landmarks -> pose + swipes, plus mouse-click polling
//! (clicks are the implicit calibration signal).

use crate::gaze::{LAND, NF};
use crate::gestures::{classify, openness, HeadShake, Pose, Swipe, SwipeDetector};
use crate::halo::LeftHand;
use crate::now;
use crate::vision::{FaceMesh, HandTracker, IrisModel, Rgb};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct Frame {
    pub t: f64,
    pub features: Option<[f64; NF]>,
    /// Right hand: gestures (voice typing, send, accept, swipes).
    pub pose: Option<Pose>,
    /// Left hand: halo size and freeze.
    pub left: Option<LeftHand>,
    /// Raw face landmarks (478 x [x, y, z], iris centres refined) + [w, h]: saved with calibrations for offline tuning.
    pub raw: Option<Arc<Vec<f32>>>,
    /// JPEG of the face box, only while recording (calibration): lets other gaze models be evaluated offline.
    pub crop: Option<Arc<Vec<u8>>>,
}

/// One UniGaze result: gaze angles for the frame captured at `t`, plus that frame's landmark features
/// (so the fast landmark model can carry the estimate forward until the next UniGaze result).
#[derive(Clone)]
pub struct UgSample {
    pub t: f64,
    pub angles: [f64; 2],
    pub land: Option<[f64; NF]>,
}

/// Small webcam image for the calibration's self view (not mirrored), with the 478 face mesh points (iris included)
/// in image fractions (0..1).
pub struct Preview {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
    pub face: Option<Vec<[f32; 2]>>,
    pub seq: u64,
}

#[derive(Default)]
pub struct Shared {
    pub latest: Option<Frame>,
    pub history: VecDeque<Frame>,
    pub ug: VecDeque<UgSample>,
    pub ug_enabled: bool,
    pub fps: f32,
    /// Self view, produced while recording (calibration).
    pub preview: Option<Arc<Preview>>,
    /// Frames per second the hands thread keeps up with.
    pub hand_fps: f32,
    pub error: Option<String>,
}

pub struct Tracker {
    pub shared: Arc<Mutex<Shared>>,
    pub swipes: Receiver<(f64, Swipe)>,
    /// Times of head shakes ("no").
    pub shakes: Receiver<f64>,
    /// (t, x, y) of left-button presses, global logical coordinates.
    pub clicks: Receiver<(f64, f64, f64)>,
    pub use_hands: Arc<AtomicBool>,
    pub mirrored: Arc<AtomicBool>,
    pub record: Arc<AtomicBool>,
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

/// 20 landmark features (then the eye patches, see `with_eye_patches`): iris offsets + eyelid openness for both eyes, head pose (3D axes + 2D geometry), face position/scale;
/// then the metric head pose from PnP for the geometric gaze model (`geo`): head rotation vector, eye position (mm).
pub fn features(p: &[[f32; 3]], w: usize, h: usize) -> [f64; NF] {
    features_at(p, w, h, crate::unigaze::focal())
}

/// Same, with a given camera focal length (px at 1280 px width): recomputing samples after a calibration estimated it.
pub fn features_at(p: &[[f32; 3]], w: usize, h: usize, focal: f64) -> [f64; NF] {
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
    let (hr, eye) = crate::unigaze::head_and_eyes(w, h, p, focal);
    let rv = hr.scaled_axis();
    let land = [
        rx, ry, ro, lx, ly, lo,
        pitch3 as f64, yaw3 as f64, roll as f64,
        cx as f64, cy as f64, (face_w / w as f32) as f64,
        yaw as f64, pitch as f64,
        rv.x, rv.y, rv.z, eye.x, eye.y, eye.z,
    ];
    let mut f = [0.0; NF];
    f[..LAND].copy_from_slice(&land);
    f // eye patches (LAND..NF) need the image: `with_eye_patches`
}

/// Fill the eye patch features (see `vision::eye_patches`).
pub fn with_eye_patches(mut f: [f64; NF], img: &Rgb, mesh: &[[f32; 3]]) -> [f64; NF] {
    for (k, v) in crate::vision::eye_patches(img, mesh).into_iter().enumerate() {
        f[LAND + k] = v as f64;
    }
    f
}

impl Tracker {
    pub fn start(camera: u32, models: std::path::PathBuf) -> Tracker {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (swipe_tx, swipes) = unbounded();
        let (shake_tx, shakes) = unbounded();
        let (click_tx, clicks) = unbounded();
        let use_hands = Arc::new(AtomicBool::new(true));
        let mirrored = Arc::new(AtomicBool::new(false));
        let record = Arc::new(AtomicBool::new(false));
        let t = Tracker {
            shared: shared.clone(),
            swipes,
            shakes,
            clicks,
            use_hands: use_hands.clone(),
            mirrored: mirrored.clone(),
            record: record.clone(),
        };
        std::thread::Builder::new()
            .name("tracker".into())
            .spawn(move || {
                if let Err(e) = run(camera, &models, &shared, swipe_tx, shake_tx, use_hands, mirrored, record) {
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

    /// (time, features, raw landmarks) of the frames with a face in (t0, t1].
    pub fn samples_between(&self, t0: f64, t1: f64) -> Vec<(f64, [f64; NF], Option<Arc<Vec<f32>>>)> {
        self.shared.lock().history.iter().filter(|f| f.t > t0 && f.t <= t1).filter_map(|f| Some((f.t, f.features?, f.raw.clone()))).collect()
    }

    pub fn raw_between(&self, t0: f64, t1: f64) -> Vec<(Arc<Vec<f32>>, Option<Arc<Vec<u8>>>)> {
        self.shared
            .lock()
            .history
            .iter()
            .filter(|f| f.t >= t0 && f.t <= t1)
            .filter_map(|f| f.raw.clone().map(|r| (r, f.crop.clone())))
            .collect()
    }

    pub fn ug_latest(&self) -> Option<UgSample> {
        self.shared.lock().ug.back().cloned()
    }

    pub fn ug_between(&self, t0: f64, t1: f64) -> Vec<UgSample> {
        self.shared.lock().ug.iter().filter(|u| u.t >= t0 && u.t <= t1).cloned().collect()
    }

    /// UniGaze results per second over the last 3 s (0 when unavailable).
    pub fn ug_rate(&self) -> f64 {
        let s = self.shared.lock();
        let now = now();
        s.ug.iter().filter(|u| now - u.t < 3.0).count() as f64 / 3.0
    }

    pub fn features_between(&self, t0: f64, t1: f64) -> Vec<[f64; NF]> {
        self.shared.lock().history.iter().filter(|f| f.t >= t0 && f.t <= t1).filter_map(|f| f.features).collect()
    }
}

fn run(
    camera: u32,
    models: &std::path::Path,
    shared_arc: &Arc<Mutex<Shared>>,
    swipe_tx: Sender<(f64, Swipe)>,
    shake_tx: Sender<f64>,
    use_hands: Arc<AtomicBool>,
    mirrored: Arc<AtomicBool>,
    record: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let shared: &Mutex<Shared> = shared_arc;
    let mut face = FaceMesh::new(models)?;
    // UniGaze on its own thread, fed with the newest frame whenever it is free (~10 Hz)
    let (ug_tx, ug_rx) = crossbeam_channel::bounded::<(f64, Vec<f32>, Option<[f64; NF]>)>(1);
    let ug_path = crate::unigaze::UniGaze::path(models);
    if ug_path.exists() {
        let shared_ug = shared_arc.clone();
        let models = models.to_path_buf();
        std::thread::Builder::new()
            .name("unigaze".into())
            .spawn(move || {
                let mut ug = match crate::unigaze::UniGaze::new(&models, 3) {
                    Ok(u) => u,
                    Err(e) => return eprintln!("UniGaze unavailable: {e:#}"),
                };
                shared_ug.lock().ug_enabled = true;
                for (t, chw, land) in ug_rx {
                    match ug.run(chw) {
                        Ok((p, y)) => {
                            let mut s = shared_ug.lock();
                            s.ug.push_back(UgSample { t, angles: [p as f64, y as f64], land });
                            while s.ug.len() > 200 {
                                s.ug.pop_front();
                            }
                        }
                        Err(e) => eprintln!("UniGaze: {e:#}"),
                    }
                }
            })?;
    } else {
        eprintln!("UniGaze model not found ({}): run scripts/fetch_unigaze.sh for better accuracy", ug_path.display());
    }
    let mut iris = IrisModel::new(models).map_err(|e| eprintln!("iris model unavailable ({e:#}): using face-mesh iris")).ok();
    let cam = Cam::open(camera).map_err(|e| {
        anyhow::anyhow!("{e:#}. Grant Camera access to your terminal app (macOS: System Settings > Privacy & Security > Camera)")
    })?;
    // hands run in parallel with the face, on every frame they can keep up with (latest frame only)
    let hand_state = Arc::new(Mutex::new(HandState::default()));
    let (job_tx, job_rx) = bounded::<HandJob>(1);
    // face centre x (image px, f32 bits; NaN: no face): the hands thread tells left from right with it
    let face_x = Arc::new(std::sync::atomic::AtomicU32::new(f32::NAN.to_bits()));
    {
        let (models, state, shared, face_x) = (models.to_path_buf(), hand_state.clone(), shared_arc.clone(), face_x.clone());
        std::thread::Builder::new().name("hands".into()).spawn(move || {
            if let Err(e) = run_hands(&models, job_rx, &state, &shared, swipe_tx, mirrored, &face_x) {
                shared.lock().error = Some(format!("{e:#}"));
            }
        })?;
    }
    let mut shake = HeadShake::new();
    let (mut t_fps, mut frames) = (now(), 0u32);
    loop {
        let img = cam.frame()?;
        let t = now();
        let (w, h) = (img.width, img.height);
        let data = Arc::new(img.data);
        let hands_on = use_hands.load(Ordering::Relaxed);
        if hands_on {
            let _ = job_tx.try_send((t, w, h, data.clone())); // busy: it gets a later frame
        }
        let rgb = Rgb { w, h, data: &data };
        let mut mesh = face.process(&rgb)?;
        if let (Some(p), Some(iris)) = (mesh.as_mut(), iris.as_mut()) {
            let [r, l] = iris.process(&rgb, p)?;
            p[468][..2].copy_from_slice(&r);
            p[473][..2].copy_from_slice(&l);
        }
        let feats = mesh.as_ref().map(|p| with_eye_patches(features(p, w, h), &rgb, p));
        let fx = mesh.as_ref().map_or(f32::NAN, |p| (p[234][0] + p[454][0]) / 2.0);
        face_x.store(fx.to_bits(), Ordering::Relaxed);
        if shake.update(t, feats.map(|f| f[12] as f32)) {
            let _ = shake_tx.send(t); // 2D yaw: nose offset / face width
        }
        if let Some(p) = mesh.as_ref() {
            if ug_tx.is_empty() {
                if let Some(chw) = crate::unigaze::normalize(&rgb, p) {
                    let _ = ug_tx.try_send((t, chw, feats));
                }
            }
        }
        if record.load(Ordering::Relaxed) && frames % 2 == 0 {
            // self view for the calibration: 1/4 size (nearest pixel: cheap), every other frame
            let (pw, ph) = (w / 4, h / 4);
            let mut rgb = Vec::with_capacity(pw * ph * 3);
            for y in 0..ph {
                let row = y * 4 * w;
                for x in 0..pw {
                    let i = (row + x * 4) * 3;
                    rgb.extend_from_slice(&data[i..i + 3]);
                }
            }
            let face = mesh.as_ref().map(|p| p.iter().map(|q| [q[0] / w as f32, q[1] / h as f32]).collect());
            let seq = shared.lock().preview.as_ref().map_or(0, |p| p.seq + 1);
            shared.lock().preview = Some(Arc::new(Preview { w: pw, h: ph, rgb, face, seq }));
        }
        let crop = match (&mesh, record.load(Ordering::Relaxed)) {
            (Some(p), true) => face_jpeg(&data, w, h, p).map(Arc::new),
            _ => None,
        };
        let raw = mesh.map(|p| {
            let mut v: Vec<f32> = p.iter().flatten().copied().collect();
            v.extend([w as f32, h as f32]);
            Arc::new(v)
        });

        let (pose, left) = if hands_on {
            let hs = hand_state.lock();
            (hs.pose, hs.left)
        } else {
            (None, None)
        };
        let f = Frame { t, features: feats, pose, left, raw, crop };
        {
            let mut s = shared.lock();
            s.history.push_back(f.clone());
            while s.history.len() > 300 {
                s.history.pop_front();
            }
            s.latest = Some(f);
        }
        frames += 1;
        if t - t_fps >= 1.0 {
            shared.lock().fps = frames as f32 / (t - t_fps) as f32;
            frames = 0;
            t_fps = t;
        }
    }
}

/// Latest hand results, written by the hands thread.
#[derive(Default)]
struct HandState {
    pose: Option<Pose>,
    left: Option<LeftHand>,
}

type HandJob = (f64, usize, usize, Arc<Vec<u8>>);

/// Hands thread: right hand -> pose + swipes, left hand -> halo control.
fn run_hands(
    models: &std::path::Path,
    jobs: Receiver<HandJob>,
    state: &Mutex<HandState>,
    shared: &Mutex<Shared>,
    swipe_tx: Sender<(f64, Swipe)>,
    mirrored: Arc<AtomicBool>,
    face_x: &std::sync::atomic::AtomicU32,
) -> anyhow::Result<()> {
    let mut hands = HandTracker::new(models)?;
    let mut swipe = SwipeDetector::new();
    let (mut t_fps, mut frames) = (now(), 0u32);
    let mut idle = false;
    for (t, w, h, data) in jobs {
        // no hand in view: every other frame is enough to notice one appearing (saves CPU)
        if std::mem::replace(&mut idle, false) {
            continue;
        }
        frames += 1;
        if t - t_fps >= 1.0 {
            shared.lock().hand_fps = frames as f32 / (t - t_fps) as f32;
            (t_fps, frames) = (t, 0);
        }
        let mirror = mirrored.load(Ordering::Relaxed);
        let mut found = hands.process(&Rgb { w, h, data: &data }, mirror)?;
        // Which hand is which: by its side of the face (the camera image is not mirrored: the user's right hand shows
        // on the image's left), which is what the user means by "left" and "right"; MediaPipe's handedness classifier
        // proved unreliable here. It is kept only when both hands sit on the same side (crossed arms) or no face.
        let fx = f32::from_bits(face_x.load(Ordering::Relaxed));
        if fx.is_finite() {
            let by_side: Vec<bool> = found.iter().map(|hd| (hd.pts[0][0] < fx) != mirror).collect();
            if !(found.len() == 2 && by_side[0] == by_side[1]) {
                for (hd, r) in found.iter_mut().zip(by_side) {
                    hd.right = r;
                }
            }
        }
        let norm = |right: bool| -> Option<Vec<[f32; 2]>> {
            let hand = found.iter().find(|x| x.right == right)?;
            Some(hand.pts.iter().map(|q| [q[0] / h as f32, q[1] / h as f32]).collect())
        };
        let (r, l) = (norm(true), norm(false));
        idle = found.is_empty();
        {
            let mut s = state.lock();
            s.pose = r.as_ref().map(|p| classify(p));
            s.left = l.as_ref().map(|p| LeftHand { wrist: p[0], openness: openness(p), raised: raised(p) });
        }
        swipe.mirrored = mirror;
        if let Some(dir) = swipe.update(t, r.as_ref().map(|p| p[0])) {
            let _ = swipe_tx.send((t, dir));
        }
    }
    Ok(())
}

/// Hand held up in front of the camera: wrist inside the frame and fingers pointing up (wrist -> middle knuckle
/// within 45 degrees of vertical). A hand resting on the desk or keyboard is low and seen flat or sideways.
/// `p`: landmarks in frame-height units.
fn raised(p: &[[f32; 2]]) -> bool {
    let (dx, dy) = (p[9][0] - p[0][0], p[9][1] - p[0][1]);
    p[0][1] < 0.95 && -dy > dx.abs()
}

/// JPEG (q90) of the face box, padded by 10%.
fn face_jpeg(rgb: &[u8], w: usize, h: usize, mesh: &[[f32; 3]]) -> Option<Vec<u8>> {
    let (bx, by, bw, bh) = crate::vision::face_box(mesh);
    let (px, py) = (bw * 0.1, bh * 0.1);
    let x0 = (bx - px).max(0.0) as usize;
    let y0 = (by - py).max(0.0) as usize;
    let x1 = ((bx + bw + px) as usize).min(w);
    let y1 = ((by + bh + py) as usize).min(h);
    if x1 <= x0 + 8 || y1 <= y0 + 8 {
        return None;
    }
    let mut buf = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
    for y in y0..y1 {
        buf.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1) * 3]);
    }
    let img = image::RgbImage::from_raw((x1 - x0) as u32, (y1 - y0) as u32, buf)?;
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode_image(&img).ok()?;
    Some(out)
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
        use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
        // 1280x720 at 30 fps if the camera has it (MJPEG): the highest frame rate alone picks 640x480 on most
        // webcams, where an eye is ~15 px wide and gaze moves the iris by less than a pixel.
        let want = CameraFormat::new(Resolution::new(1280, 720), FrameFormat::MJPEG, 30);
        let requests = [RequestedFormatType::Closest(want), RequestedFormatType::AbsoluteHighestFrameRate];
        let mut last = None;
        for req in requests {
            match nokhwa::Camera::new(CameraIndex::Index(index), RequestedFormat::new::<RgbFormat>(req)).and_then(|mut cam| {
                cam.open_stream()?;
                Ok(cam)
            }) {
                Ok(cam) => {
                    println!("camera: {}", cam.camera_format());
                    return Ok(Self(std::sync::Mutex::new(cam)));
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.map_or_else(|| anyhow::anyhow!("no camera"), Into::into))
    }

    fn frame(&self) -> anyhow::Result<RgbFrame> {
        use nokhwa::pixel_format::RgbFormat;
        let img = self.0.lock().unwrap().frame()?.decode_image::<RgbFormat>()?;
        Ok(RgbFrame { width: img.width() as usize, height: img.height() as usize, data: img.into_raw() })
    }
}

/// Mouse-button polling: every left click is a free calibration sample (people look at what they click).
/// macOS: Quartz button state + cursor position, which need no Accessibility permission. Linux: device_query (X11).
fn poll_clicks(tx: Sender<(f64, f64, f64)>) {
    let mut was_down = false;
    #[cfg(not(target_os = "macos"))]
    let state = device_query::DeviceState::new();
    loop {
        #[cfg(target_os = "macos")]
        let (down, x, y) = mac_mouse();
        #[cfg(not(target_os = "macos"))]
        let (down, x, y) = {
            use device_query::DeviceQuery;
            let m = state.get_mouse();
            (m.button_pressed.get(1).copied().unwrap_or(false), m.coords.0 as f64, m.coords.1 as f64)
        };
        if down && !was_down {
            let _ = tx.send((now(), x, y));
        }
        was_down = down;
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
}

#[cfg(target_os = "macos")]
fn mac_mouse() -> (bool, f64, f64) {
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceButtonState(state: i32, button: u32) -> bool;
    }
    // kCGEventSourceStateCombinedSessionState = 0, kCGMouseButtonLeft = 0
    let down = unsafe { CGEventSourceButtonState(0, 0) };
    let pos = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .ok()
        .and_then(|src| CGEvent::new(src).ok())
        .map(|e| e.location());
    match pos {
        Some(p) => (down, p.x, p.y), // global, top-left origin, points
        None => (false, 0.0, 0.0),
    }
}
