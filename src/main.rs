//! precog: webcam gaze tracking + hand gestures + voice typing + LLM intent suggestions.

mod actions;
mod agent;
mod cloud;
#[cfg(target_os = "macos")]
mod camera_mac;
mod dictation;
mod gaze;
mod gestures;
mod llm;
mod monitors;
mod paths;
mod settings;
mod tracker;
mod tray;
mod unigaze;
mod ui;
mod vision;

use clap::Parser;
use std::sync::OnceLock;
use std::time::Instant;

/// Seconds since start (monotonic), shared by every thread.
pub fn now() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[derive(Parser)]
#[command(name = "precog", about = "Gaze-aware proactive assistant")]
struct Args {
    /// Force a new calibration
    #[arg(long)]
    calibrate: bool,
    /// Camera index
    #[arg(long, default_value_t = 0)]
    camera: u32,
    /// Gaze tracking only, no LLM suggestions (this session)
    #[arg(long)]
    no_agent: bool,
    /// No hand gestures (this session)
    #[arg(long)]
    no_gesture: bool,
    /// Hide the gaze ring (this session)
    #[arg(long)]
    no_dot: bool,
    /// Don't learn from mouse clicks (this session)
    #[arg(long)]
    no_learn: bool,
    /// No menu-bar / tray icon
    #[arg(long)]
    no_tray: bool,
    /// One screenshot -> LLM -> print the suggestion, then exit
    #[arg(long)]
    test_llm: bool,
    /// Debug: run face + hand models on an image and print landmarks as JSON
    #[arg(long, hide = true)]
    dump_landmarks: Option<std::path::PathBuf>,
    /// Debug: time each vision stage on an image
    #[arg(long, hide = true)]
    bench: Option<std::path::PathBuf>,
    /// Debug: UniGaze on an image: print (pitch, yaw), write the normalised face next to it
    #[arg(long, hide = true)]
    unigaze: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    now();
    let args = Args::parse();
    if let Some(path) = &args.unigaze {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = vision::Rgb { w, h, data: img.as_raw() };
        let dir = paths::onnx_dir();
        let mesh = vision::FaceMesh::new(&dir)?.process(&rgb)?.ok_or_else(|| anyhow::anyhow!("no face"))?;
        let chw = unigaze::normalize(&rgb, &mesh).ok_or_else(|| anyhow::anyhow!("normalisation failed"))?;
        let n = unigaze::SIZE * unigaze::SIZE;
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let out = image::RgbImage::from_fn(unigaze::SIZE as u32, unigaze::SIZE as u32, |x, y| {
            let i = y as usize * unigaze::SIZE + x as usize;
            image::Rgb([0, 1, 2].map(|c| ((chw[c * n + i] * std[c] + mean[c]) * 255.0).clamp(0.0, 255.0) as u8))
        });
        out.save(path.with_extension("norm.png"))?;
        let mut ug = unigaze::UniGaze::new(&dir, 4)?;
        let t0 = now();
        let (p, y) = ug.run(chw.clone())?;
        let dt = now() - t0;
        let t1 = now();
        for _ in 0..5 {
            ug.run(chw.clone())?;
        }
        println!("pitch {p:.4} yaw {y:.4} rad  (first {:.0} ms, then {:.0} ms/frame)", dt * 1000.0, (now() - t1) * 200.0);
        let mesh_json: Vec<[f32; 3]> = mesh;
        std::fs::write(path.with_extension("mesh.json"), serde_json::to_vec(&mesh_json)?)?;
        return Ok(());
    }
    if let Some(path) = &args.bench {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = vision::Rgb { w, h, data: img.as_raw() };
        let dir = paths::onnx_dir();
        let (mut face, mut iris, mut hand) = (vision::FaceMesh::new(&dir)?, vision::IrisModel::new(&dir)?, vision::HandTracker::new(&dir)?);
        let p = face.process(&rgb)?.expect("no face");
        let n = 50;
        let time = |name: &str, f: &mut dyn FnMut()| {
            let t0 = now();
            for _ in 0..n {
                f();
            }
            println!("{name:>14}: {:6.2} ms", (now() - t0) * 1000.0 / n as f64);
        };
        time("face (tracked)", &mut || {
            face.process(&rgb).unwrap();
        });
        time("iris (2 eyes)", &mut || {
            iris.process(&rgb, &p).unwrap();
        });
        time("hand", &mut || {
            hand.process(&rgb).unwrap();
        });
        time("hand (reset)", &mut || {
            hand = vision::HandTracker::new(&dir).unwrap();
            hand.process(&rgb).unwrap();
        });
        return Ok(());
    }
    if let Some(path) = &args.dump_landmarks {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = vision::Rgb { w, h, data: img.as_raw() };
        let dir = paths::onnx_dir();
        let face = vision::FaceMesh::new(&dir)?.process(&rgb)?;
        let hand = vision::HandTracker::new(&dir)?.process(&rgb)?;
        let iris = match &face {
            Some(p) => Some(vision::IrisModel::new(&dir)?.process(&rgb, p)?),
            None => None,
        };
        let feats = face.as_ref().map(|p| tracker::features(p, w, h).to_vec());
        let pose = hand.as_ref().map(|p| format!("{:?}", gestures::classify(&p.iter().map(|q| [q[0], q[1]]).collect::<Vec<_>>())));
        println!("{}", serde_json::json!({"face": face, "hand": hand, "iris": iris, "features": feats, "pose": pose}));
        return Ok(());
    }
    if args.test_llm {
        let cfg = settings::Settings::load();
        let mon = monitors::list()[0];
        let (cx, cy) = (mon.left as f64 + mon.width as f64 / 2.0, mon.top as f64 + mon.height as f64 / 2.0);
        let jpeg = agent::screenshot_with_gaze(&[(now(), cx, cy)], &mon, now())?;
        let (p, m) = llm::resolve(&cfg);
        println!("provider/model: {} {m}", p.id);
        let t0 = now();
        let a = agent::ask(&cfg, &jpeg, &actions::frontmost_app(), &[])?;
        println!("{}\n({:.1}s)", serde_json::to_string_pretty(&a)?, now() - t0);
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        // ask for camera permission up front (shows the system prompt the first time)
        let (tx, rx) = std::sync::mpsc::channel();
        nokhwa::nokhwa_initialize(move |granted| {
            let _ = tx.send(granted);
        });
        if !rx.recv_timeout(std::time::Duration::from_secs(60)).unwrap_or(false) {
            anyhow::bail!("camera access denied: allow it for your terminal in System Settings > Privacy & Security > Camera");
        }
    }

    let mut overrides = Vec::new();
    for (flag, key) in [(args.no_agent, "agent"), (args.no_gesture, "gesture"), (args.no_dot, "dot"), (args.no_learn, "learn")] {
        if flag {
            overrides.push((key, false));
        }
    }
    let opts = ui::Options { camera: args.camera, calibrate: args.calibrate, no_tray: args.no_tray, overrides };

    let primary = monitors::list()[0];
    let mut native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("precog")
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_taskbar(false)
            .with_active(false)
            .with_resizable(false)
            .with_inner_size([720.0, 52.0])
            .with_position([primary.left as f32 + 12.0, primary.bottom() as f32 - 70.0]),
        ..Default::default()
    };
    #[cfg(target_os = "macos")]
    {
        native.event_loop_builder = Some(Box::new(|b| {
            use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
            b.with_activation_policy(ActivationPolicy::Accessory); // menu-bar app: no Dock icon
        }));
    }
    #[cfg(not(target_os = "macos"))]
    let _ = &mut native;
    eframe::run_native("precog", native, Box::new(move |cc| Ok(Box::new(ui::PrecogApp::new(cc, opts)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}
