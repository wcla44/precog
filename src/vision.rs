//! MediaPipe face / hand pipelines on ONNX Runtime (models converted from the official .task bundles).
//!
//! detector (SSD + anchors) -> rotated square ROI -> landmark model -> landmarks back in image pixels.
//! Pre/post-processing mirrors MediaPipe and was validated against it (face: ~0.6 px mean error).

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::f32::consts::PI;
use std::path::Path;

/// RGB8 image, row-major.
pub struct Rgb<'a> {
    pub w: usize,
    pub h: usize,
    pub data: &'a [u8],
}

/// dst pixel (u, v) -> src pixel: x = a*u + b*v + c, y = d*u + e*v + f
#[derive(Clone, Copy, Debug)]
pub struct Affine([f32; 6]);

impl Affine {
    pub fn apply(&self, u: f32, v: f32) -> (f32, f32) {
        let m = self.0;
        (m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5])
    }
}

/// Bilinear warp into an NHWC float tensor buffer. `norm` maps a u8 to the model's input range.
fn warp(img: &Rgb, m: Affine, out: usize, norm: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut buf = vec![0f32; out * out * 3];
    let zero = norm(0.0);
    for v in 0..out {
        for u in 0..out {
            let (x, y) = m.apply(u as f32, v as f32);
            let o = (v * out + u) * 3;
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let px = |xx: isize, yy: isize, c: usize| -> f32 {
                if xx < 0 || yy < 0 || xx >= img.w as isize || yy >= img.h as isize {
                    0.0
                } else {
                    img.data[(yy as usize * img.w + xx as usize) * 3 + c] as f32
                }
            };
            if x0 < -1 || y0 < -1 || x0 >= img.w as isize || y0 >= img.h as isize {
                buf[o..o + 3].fill(zero);
                continue;
            }
            for c in 0..3 {
                let a = px(x0, y0, c) * (1.0 - fx) + px(x0 + 1, y0, c) * fx;
                let b = px(x0, y0 + 1, c) * (1.0 - fx) + px(x0 + 1, y0 + 1, c) * fx;
                buf[o + c] = norm(a * (1.0 - fy) + b * fy);
            }
        }
    }
    buf
}

fn anchors(size: usize, strides: &[usize], per_layer: &[usize]) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for (&stride, &n) in strides.iter().zip(per_layer) {
        let g = size.div_ceil(stride);
        for y in 0..g {
            for x in 0..g {
                for _ in 0..n {
                    out.push(((x as f32 + 0.5) / g as f32, (y as f32 + 0.5) / g as f32));
                }
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-100.0, 100.0)).exp())
}

fn norm_angle(a: f32) -> f32 {
    a - 2.0 * PI * ((a + PI) / (2.0 * PI)).floor()
}

struct Detection {
    score: f32,
    center: (f32, f32),
    size: (f32, f32),
    keypoints: Vec<(f32, f32)>,
}

struct Detector {
    session: Session,
    input: String,
    size: usize,
    anchors: Vec<(f32, f32)>,
    keypoints: usize,
    signed: bool, // input range [-1, 1] instead of [0, 1]
}

impl Detector {
    /// Detect in the whole frame (letterboxed), like MediaPipe.
    fn detect(&mut self, img: &Rgb) -> Result<Option<Detection>> {
        let s = img.w.max(img.h) as f32;
        self.detect_in(img, -(s - img.w as f32) / 2.0, -(s - img.h as f32) / 2.0, s)
    }

    /// Detect inside the square [x0, x0+side] x [y0, y0+side] (image pixels, may extend past the borders).
    fn detect_in(&mut self, img: &Rgb, x0: f32, y0: f32, side: f32) -> Result<Option<Detection>> {
        // half-pixel centres like cv2.resize
        let k = side / self.size as f32;
        let m = Affine([k, 0.0, 0.5 * k - 0.5 + x0, 0.0, k, 0.5 * k - 0.5 + y0]);
        let signed = self.signed;
        let data = warp(img, m, self.size, |p| if signed { p / 127.5 - 1.0 } else { p / 255.0 });
        let tensor = Tensor::from_array(([1usize, self.size, self.size, 3], data))?;
        let outputs = self.session.run(ort::inputs![self.input.as_str() => tensor])?;
        let (_, reg) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, cls) = outputs[1].try_extract_tensor::<f32>()?;
        let (best, &raw) = cls
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("empty detector output")?;
        let score = sigmoid(raw);
        if score < 0.5 {
            return Ok(None);
        }
        let stride = reg.len() / cls.len();
        let r = &reg[best * stride..(best + 1) * stride];
        let sz = self.size as f32;
        let a = self.anchors[best];
        let to_img = |px: f32, py: f32| (px * side + x0, py * side + y0);
        let center = to_img(r[0] / sz + a.0, r[1] / sz + a.1);
        let keypoints = (0..self.keypoints)
            .map(|i| to_img(r[4 + 2 * i] / sz + a.0, r[5 + 2 * i] / sz + a.1))
            .collect();
        Ok(Some(Detection { score, center, size: (r[2] / sz * side, r[3] / sz * side), keypoints }))
    }

    /// Whole frame first; if nothing, zoom on central squares. The short-range face model needs the face to be
    /// ~25+ px in its 128 px input, i.e. a face under ~250 px wide in a 1280 px frame is often missed otherwise.
    fn detect_multiscale(&mut self, img: &Rgb) -> Result<Option<Detection>> {
        if let Some(d) = self.detect(img)? {
            return Ok(Some(d));
        }
        let (w, h) = (img.w as f32, img.h as f32);
        let m = w.min(h);
        for (frac, cy) in [(1.0, 0.5), (0.6, 0.45), (0.6, 0.3)] {
            let side = m * frac;
            if let Some(d) = self.detect_in(img, (w - side) / 2.0, h * cy - side / 2.0, side)? {
                return Ok(Some(d));
            }
        }
        Ok(None)
    }
}

/// Rotated square region of interest in image pixels.
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub side: f32,
    pub angle: f32,
}

impl Roi {
    /// Affine from an `out`x`out` crop to the image.
    fn affine(&self, out: usize) -> Affine {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let k = self.side / out as f32;
        let h = out as f32 / 2.0;
        Affine([c * k, -s * k, self.cx - c * k * h + s * k * h, s * k, c * k, self.cy - s * k * h - c * k * h])
    }
}

fn session(dir: &Path, name: &str) -> Result<Session> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    let env = match ENV.get() {
        Some(e) => e,
        None => {
            let e = ort::init().with_name("precog").build().map_err(|e| anyhow::anyhow!("onnxruntime: {e}"))?;
            ENV.get_or_init(|| e)
        }
    };
    let path = dir.join(name);
    let threads = std::env::var("PRECOG_ORT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    // ort defaults to an "efficiency" device policy, which on Apple silicon is ~9x slower for these small
    // models (21 ms vs 2.4 ms for the face mesh): force the plain CPU path.
    let b = Session::builder(env)?
        .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
        .and_then(|b| b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(threads))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut b = b;
    b.commit_from_file(&path).with_context(|| format!("loading {} (run scripts/convert_models.sh)", path.display()))
}

fn input_name(s: &Session) -> String {
    s.inputs()[0].name().to_string()
}

/// 478 face landmarks (468 mesh + 10 iris) as (x, y, z) in image pixels.
pub struct FaceMesh {
    detector: Detector,
    landmarks: Session,
    input: String,
    roi: Option<Roi>,
}

impl FaceMesh {
    pub fn new(dir: &Path) -> Result<Self> {
        let det = session(dir, "face_detector.onnx")?;
        let lm = session(dir, "face_landmarks_detector.onnx")?;
        Ok(Self {
            detector: Detector {
                input: input_name(&det),
                session: det,
                size: 128,
                anchors: anchors(128, &[8, 16], &[2, 6]),
                keypoints: 6,
                signed: true,
            },
            input: input_name(&lm),
            landmarks: lm,
            roi: None,
        })
    }

    pub fn process(&mut self, img: &Rgb) -> Result<Option<Vec<[f32; 3]>>> {
        let roi = match self.roi {
            Some(r) => r,
            None => {
                let Some(d) = self.detector.detect_multiscale(img)? else { return Ok(None) };
                let _ = d.score;
                let (r, l) = (d.keypoints[0], d.keypoints[1]); // right eye -> left eye, target angle 0
                let angle = norm_angle(-(-(l.1 - r.1)).atan2(l.0 - r.0));
                Roi { cx: d.center.0, cy: d.center.1, side: d.size.0.max(d.size.1) * 1.5, angle }
            }
        };
        let m = roi.affine(256);
        let data = warp(img, m, 256, |p| p / 255.0);
        let tensor = Tensor::from_array(([1usize, 256, 256, 3], data))?;
        let outputs = self.landmarks.run(ort::inputs![self.input.as_str() => tensor])?;
        let (_, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, presence) = outputs[1].try_extract_tensor::<f32>()?;
        if sigmoid(presence[0]) < 0.5 {
            self.roi = None;
            return Ok(None);
        }
        let zk = roi.side / 256.0;
        let pts: Vec<[f32; 3]> = raw
            .chunks_exact(3)
            .map(|p| {
                let (x, y) = m.apply(p[0], p[1]);
                [x, y, p[2] * zk]
            })
            .collect();
        self.roi = Some(roi_from_points(&pts, pts[33], pts[263], 0.0, 1.5, 0.0));
        Ok(Some(pts))
    }
}

/// Next-frame ROI from landmarks: rotation from two reference points, bbox in the rotated frame.
fn roi_from_points(pts: &[[f32; 3]], a: [f32; 3], b: [f32; 3], target: f32, scale: f32, shift_y: f32) -> Roi {
    let angle = norm_angle(target - (-(b[1] - a[1])).atan2(b[0] - a[0]));
    let (c, s) = (angle.cos(), angle.sin());
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in pts {
        let (rx, ry) = (p[0] * c + p[1] * s, -p[0] * s + p[1] * c);
        x0 = x0.min(rx);
        x1 = x1.max(rx);
        y0 = y0.min(ry);
        y1 = y1.max(ry);
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let (rcx, rcy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0 + shift_y * h);
    Roi { cx: rcx * c - rcy * s, cy: rcx * s + rcy * c, side: w.max(h) * scale, angle }
}

/// 21 hand landmarks in image pixels.
pub struct HandTracker {
    detector: Detector,
    landmarks: Session,
    input: String,
    roi: Option<Roi>,
}

impl HandTracker {
    pub fn new(dir: &Path) -> Result<Self> {
        let det = session(dir, "hand_detector.onnx")?;
        let lm = session(dir, "hand_landmarks_detector.onnx")?;
        Ok(Self {
            detector: Detector {
                input: input_name(&det),
                session: det,
                size: 192,
                anchors: anchors(192, &[8, 16], &[2, 6]),
                keypoints: 7,
                signed: false,
            },
            input: input_name(&lm),
            landmarks: lm,
            roi: None,
        })
    }

    pub fn process(&mut self, img: &Rgb) -> Result<Option<Vec<[f32; 3]>>> {
        let roi = match self.roi {
            Some(r) => r,
            None => {
                let Some(d) = self.detector.detect(img)? else { return Ok(None) };
                // wrist (kp0) -> middle finger MCP (kp2), target 90 degrees
                let (w, m) = (d.keypoints[0], d.keypoints[2]);
                let angle = norm_angle(PI / 2.0 - (-(m.1 - w.1)).atan2(m.0 - w.0));
                let side = d.size.0.max(d.size.1);
                let (sx, sy) = (0.0, -0.5 * side); // shift towards the fingers
                Roi {
                    cx: d.center.0 + sx * angle.cos() - sy * angle.sin(),
                    cy: d.center.1 + sx * angle.sin() + sy * angle.cos(),
                    side: side * 2.6,
                    angle,
                }
            }
        };
        let m = roi.affine(224);
        let data = warp(img, m, 224, |p| p / 255.0);
        let tensor = Tensor::from_array(([1usize, 224, 224, 3], data))?;
        let outputs = self.landmarks.run(ort::inputs![self.input.as_str() => tensor])?;
        let (_, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, presence) = outputs[1].try_extract_tensor::<f32>()?;
        if presence[0] < 0.8 { // real hands score >0.99; the palm detector sometimes fires on faces (~0.5)
            self.roi = None;
            return Ok(None);
        }
        let pts: Vec<[f32; 3]> = raw
            .chunks_exact(3)
            .map(|p| {
                let (x, y) = m.apply(p[0], p[1]);
                [x, y, p[2]]
            })
            .collect();
        // track: wrist -> middle MCP, like MediaPipe's hand landmarks -> ROI
        self.roi = Some(roi_from_points(&pts, pts[0], pts[9], PI / 2.0, 2.0, -0.1));
        Ok(Some(pts))
    }
}

/// MediaPipe iris model: 64x64 crop per eye -> refined iris centre (5 iris points + 71 eye-contour points).
/// More precise than the face mesh's own iris points, which come from a 256x256 crop of the whole face.
/// Pre-processing found empirically against the face mesh: input [0, 1], the subject's right eye is mirrored.
pub struct IrisModel {
    session: Session,
    input: String,
}

/// (corner 1, corner 2, mirror?) for the subject's right eye, then left eye.
const IRIS_EYES: [(usize, usize, bool); 2] = [(33, 133, true), (362, 263, false)];

impl IrisModel {
    pub fn new(dir: &Path) -> Result<Self> {
        let s = session(dir, "iris_landmark.onnx")?;
        Ok(Self { input: input_name(&s), session: s })
    }

    /// Refined iris centres (right eye, left eye) in image pixels, from face-mesh eye corners.
    pub fn process(&mut self, img: &Rgb, mesh: &[[f32; 3]]) -> Result<[[f32; 2]; 2]> {
        let mut out = [[0.0; 2]; 2];
        for (k, &(c1, c2, mirror)) in IRIS_EYES.iter().enumerate() {
            let (a, b) = (mesh[c1], mesh[c2]);
            let angle = norm_angle(-(-(b[1] - a[1])).atan2(b[0] - a[0]));
            let side = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() * 2.3;
            let roi = Roi { cx: (a[0] + b[0]) / 2.0, cy: (a[1] + b[1]) / 2.0, side, angle };
            let m = roi.affine(64);
            // mirrored crop: pixel u of the input is pixel 63-u of the crop
            let mf = if mirror {
                let p = m.0;
                Affine([-p[0], p[1], p[0] * 63.0 + p[2], -p[3], p[4], p[3] * 63.0 + p[5]])
            } else {
                m
            };
            let data = warp(img, mf, 64, |p| p / 255.0);
            let tensor = Tensor::from_array(([1usize, 64, 64, 3], data))?;
            let outputs = self.session.run(ort::inputs![self.input.as_str() => tensor])?;
            let (_, iris) = outputs[1].try_extract_tensor::<f32>()?;
            let (mut x, y) = (iris[0], iris[1]); // centre of the iris
            if mirror {
                x = 64.0 - x;
            }
            let (ix, iy) = m.apply(x, y);
            out[k] = [ix, iy];
        }
        Ok(out)
    }
}

/// Tight face box from the mesh: (x0, y0, w, h).
pub fn face_box(mesh: &[[f32; 3]]) -> (f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &mesh[..468] {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    (x0, y0, x1 - x0, y1 - y0)
}
