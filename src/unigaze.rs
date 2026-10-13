//! UniGaze (ut-vision/UniGaze, ViT-B/16, int8 ONNX): appearance-based gaze from the whole face, robust to head pose.
//!
//! Pre-processing is ETH-XGaze "data normalisation": head pose from a 6-point 3D face model (PnP), then a perspective
//! warp of the face to a virtual camera looking straight at it (focal 960, 600 mm, 224x224). The model outputs
//! (pitch, yaw) in that normalised space, which is what we regress to the screen (best on real recordings).
//! Weights are fetched by scripts/fetch_unigaze.sh (non-commercial license; never committed).

use crate::vision::Rgb;
use anyhow::{Context, Result};
use nalgebra::{Matrix3, Rotation3, Vector2, Vector3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// 3D face model points (mm), ETH-XGaze face_model.txt rows [20, 23, 26, 29, 15, 19]:
/// eye corners (subject's right outer, right inner, left inner, left outer) and the two nostrils.
const FACE: [[f64; 3]; 6] = [
    [-46.509_498_596, -38.327_095_032, 36.416_004_181],
    [-17.760_721_207, -32.585_197_449, 29.076_156_616],
    [18.043_910_980, -30.956_823_349, 29.062_967_300],
    [44.737_586_975, -34.107_872_009, 36.732_437_134],
    [-10.611_664_772, 12.958_349_228, 21.622_764_587],
    [11.289_629_936, 13.864_244_461, 21.837_900_162],
];
/// Matching MediaPipe face-mesh indices.
const MP: [usize; 6] = [33, 133, 362, 263, 98, 327];
const FOCAL_N: f64 = 960.0;
const DIST_N: f64 = 600.0;
pub const SIZE: usize = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Camera focal length in px at 1280 px width (f64 bits). Default ~1000 (MacBook-class FOV); a calibration estimates
/// the real one (wide-angle webcams are around 600-700) and sets it, see `set_focal`.
static FOCAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x408F_4000_0000_0000); // 1000.0

pub fn focal() -> f64 {
    f64::from_bits(FOCAL.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn set_focal(f: f64) {
    if f.is_finite() && (200.0..4000.0).contains(&f) {
        FOCAL.store(f.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Approximate webcam intrinsics: the focal above, principal point at the centre.
pub fn camera(w: usize, h: usize) -> Matrix3<f64> {
    camera_with(w, h, focal())
}

pub fn camera_with(w: usize, h: usize, focal_1280: f64) -> Matrix3<f64> {
    let f = focal_1280 * w as f64 / 1280.0;
    Matrix3::new(f, 0.0, w as f64 / 2.0, 0.0, f, h as f64 / 2.0, 0.0, 0.0, 1.0)
}

fn project(k: &Matrix3<f64>, r: &Rotation3<f64>, t: &Vector3<f64>, p: &[f64; 3]) -> Vector2<f64> {
    let c = r * Vector3::new(p[0], p[1], p[2]) + t;
    Vector2::new(k[(0, 0)] * c.x / c.z + k[(0, 2)], k[(1, 1)] * c.y / c.z + k[(1, 2)])
}

/// Head pose (rotation, translation in mm) by Gauss-Newton on the reprojection error, from a frontal initial guess.
pub fn solve_pnp(k: &Matrix3<f64>, img_pts: &[Vector2<f64>; 6]) -> (Rotation3<f64>, Vector3<f64>) {
    let span_model = FACE[3][0] - FACE[0][0];
    let span_img = (img_pts[3] - img_pts[0]).norm().max(1.0);
    let z0 = k[(0, 0)] * span_model / span_img;
    let mean_img = img_pts.iter().fold(Vector2::zeros(), |a, p| a + p) / 6.0;
    let mean_mod = FACE.iter().fold(Vector3::zeros(), |a, p| a + Vector3::new(p[0], p[1], p[2])) / 6.0;
    let mut x = nalgebra::SVector::<f64, 6>::from_column_slice(&[
        0.0,
        0.0,
        0.0,
        (mean_img.x - k[(0, 2)]) * z0 / k[(0, 0)] - mean_mod.x,
        (mean_img.y - k[(1, 2)]) * z0 / k[(1, 1)] - mean_mod.y,
        z0 - mean_mod.z,
    ]);
    let residual = |x: &nalgebra::SVector<f64, 6>| {
        let r = Rotation3::new(Vector3::new(x[0], x[1], x[2]));
        let t = Vector3::new(x[3], x[4], x[5]);
        let mut e = nalgebra::SVector::<f64, 12>::zeros();
        for i in 0..6 {
            let p = project(k, &r, &t, &FACE[i]) - img_pts[i];
            e[2 * i] = p.x;
            e[2 * i + 1] = p.y;
        }
        e
    };
    for _ in 0..15 {
        let e = residual(&x);
        let mut j = nalgebra::SMatrix::<f64, 12, 6>::zeros();
        for c in 0..6 {
            let h = if c < 3 { 1e-6 } else { 1e-3 };
            let mut xp = x;
            xp[c] += h;
            j.set_column(c, &((residual(&xp) - e) / h));
        }
        let jt = j.transpose();
        let Some(step) = (jt * j + nalgebra::SMatrix::<f64, 6, 6>::identity() * 1e-6).cholesky().map(|c| c.solve(&(jt * e))) else {
            break;
        };
        x -= step;
        if step.norm() < 1e-7 {
            break;
        }
    }
    (Rotation3::new(Vector3::new(x[0], x[1], x[2])), Vector3::new(x[3], x[4], x[5]))
}

/// Head pose and eye position in the camera frame (mm; x right, y down, z from the camera towards the user, as
/// seen in the image): head rotation, and the midpoint of the two eyeball centres (~10 mm behind the eye corners).
pub fn head_and_eyes(w: usize, h: usize, mesh: &[[f32; 3]], focal_1280: f64) -> (Rotation3<f64>, Vector3<f64>) {
    let k = camera_with(w, h, focal_1280);
    let pts: [Vector2<f64>; 6] = MP.map(|i| Vector2::new(mesh[i][0] as f64, mesh[i][1] as f64));
    let (hr, t) = solve_pnp(&k, &pts);
    let corner = |i: usize| Vector3::new(FACE[i][0], FACE[i][1], FACE[i][2]);
    let eyes = (corner(0) + corner(1) + corner(2) + corner(3)) / 4.0 + Vector3::new(0.0, 0.0, 10.0);
    (hr, hr * eyes + t)
}

/// Normalised face (CHW, ImageNet-normalised RGB, 224x224) from a frame and its face mesh.
pub fn normalize(img: &Rgb, mesh: &[[f32; 3]]) -> Option<Vec<f32>> {
    let k = camera(img.w, img.h);
    let pts: [Vector2<f64>; 6] = MP.map(|i| Vector2::new(mesh[i][0] as f64, mesh[i][1] as f64));
    let (hr, t) = solve_pnp(&k, &pts);
    let fc: Vec<Vector3<f64>> = FACE.iter().map(|p| hr * Vector3::new(p[0], p[1], p[2]) + t).collect();
    let eyes = (fc[0] + fc[1] + fc[2] + fc[3]) / 4.0;
    let nose = (fc[4] + fc[5]) / 2.0;
    let center = (eyes + nose) / 2.0;
    let dist = center.norm();
    if !(100.0..3000.0).contains(&dist) {
        return None;
    }
    let fwd = center / dist;
    let down = fwd.cross(&hr.matrix().column(0)).normalize();
    let right = down.cross(&fwd).normalize();
    let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd.transpose()]);
    let s = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, DIST_N / dist));
    let cam_n = Matrix3::new(FOCAL_N, 0.0, SIZE as f64 / 2.0, 0.0, FOCAL_N, SIZE as f64 / 2.0, 0.0, 0.0, 1.0);
    let w = cam_n * s * r * k.try_inverse()?;
    let winv = w.try_inverse()?; // normalised pixel -> frame pixel
    let n = SIZE * SIZE;
    let mut out = vec![0f32; 3 * n];
    for v in 0..SIZE {
        for u in 0..SIZE {
            let p = winv * Vector3::new(u as f64, v as f64, 1.0);
            let (x, y) = ((p.x / p.z) as f32, (p.y / p.z) as f32);
            let rgb = bilinear(img, x, y);
            for c in 0..3 {
                out[c * n + v * SIZE + u] = (rgb[c] / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Some(out)
}

fn bilinear(img: &Rgb, x: f32, y: f32) -> [f32; 3] {
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
    let mut o = [0f32; 3];
    for (c, v) in o.iter_mut().enumerate() {
        let a = px(x0, y0, c) * (1.0 - fx) + px(x0 + 1, y0, c) * fx;
        let b = px(x0, y0 + 1, c) * (1.0 - fx) + px(x0 + 1, y0 + 1, c) * fx;
        *v = a * (1.0 - fy) + b * fy;
    }
    o
}

pub struct UniGaze {
    session: Session,
}

impl UniGaze {
    pub fn path(dir: &Path) -> std::path::PathBuf {
        dir.join("unigaze_b16_int8.onnx")
    }

    pub fn new(dir: &Path, threads: usize) -> Result<Self> {
        let p = Self::path(dir);
        anyhow::ensure!(p.exists(), "{} missing: run scripts/fetch_unigaze.sh", p.display());
        Ok(Self { session: crate::vision::session_with_threads(&p, threads).context("loading UniGaze")? })
    }

    /// (pitch, yaw) in radians, normalised camera space.
    pub fn run(&mut self, chw: Vec<f32>) -> Result<(f32, f32)> {
        let tensor = Tensor::from_array(([1usize, 3, SIZE, SIZE], chw))?;
        let out = self.session.run(ort::inputs!["image" => tensor])?;
        let (_, v) = out[0].try_extract_tensor::<f32>()?;
        Ok((v[0], v[1]))
    }
}
