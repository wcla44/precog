//! egui app: HUD (root window), gaze ring, multi-monitor calibration, suggestion popup, voice overlay,
//! settings window. Core logic runs in `logic()` every frame; drawing in `ui()`.

use crate::actions::{self, Action};
use crate::agent::Agent;
use crate::cloud::{self, Smudge};
use crate::dictation::{self, Dictation};
use crate::gaze::{ClickCorrector, FixationFilter, GazeModel, CLICK_GROUP, FEATURE_VERSION, NF};
use crate::gestures::{Debounce, Pose, Swipe};
use crate::monitors::{self, Monitor};
use crate::settings::Settings;
use crate::tracker::Tracker;
use crate::tray::{self, Tray};
use crate::{llm, now, paths};
use egui::{Align2, Color32, CornerRadius, FontId, Frame, Margin, Pos2, ViewportBuilder, ViewportCommand, ViewportId};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const BG: Color32 = Color32::from_rgb(28, 28, 30);
const CODE_BG: Color32 = Color32::from_rgb(13, 17, 23);
const GREEN: Color32 = Color32::from_rgb(126, 231, 135);
const RED: Color32 = Color32::from_rgb(255, 59, 48);

// ---------------------------------------------------------------------------------------------
struct Calibration {
    points: Vec<(usize, f64, f64)>, // (monitor index, x, y) global
    i: Option<usize>,
    t0: f64,
    settle: f64,
    model: GazeModel,
    umodel: GazeModel,
    /// Raw landmarks per dot, saved for offline tuning of features / models.
    raw: Vec<serde_json::Value>,
    /// Dots already re-queued once because the face was not seen.
    retried: std::collections::HashSet<(usize, i64, i64)>,
}

impl Calibration {
    // short on purpose: clicks keep refining the model afterwards
    const SETTLE: f64 = 0.45;
    const SWITCH_SETTLE: f64 = 1.0;
    const COLLECT: f64 = 0.65;

    fn new(mons: &[Monitor]) -> Self {
        let mut points = Vec::new();
        for (mi, m) in mons.iter().enumerate() {
            // 3x3 per screen in a snake path: predictable, so the eyes get there fast
            let pts = (0..9).map(|k| {
                let (row, col) = (k / 3, if (k / 3) % 2 == 0 { k % 3 } else { 2 - k % 3 });
                let (fx, fy) = (0.04 + 0.46 * col as f64, 0.04 + 0.46 * row as f64); // close to the edges
                (mi, m.left as f64 + fx * m.width as f64, m.top as f64 + fy * m.height as f64)
            });
            points.extend(pts);
        }
        let mut model = GazeModel::landmarks();
        model.layout = mons.iter().map(|m| m.layout()).collect();
        let mut umodel = GazeModel::unigaze();
        umodel.layout = model.layout.clone();
        Self { points, i: None, t0: 0.0, settle: Self::SWITCH_SETTLE, model, umodel, raw: Vec::new(), retried: Default::default() }
    }

    fn start_point(&mut self, i: usize, t: f64) {
        let switched = i == 0 || self.points[i].0 != self.points[i - 1].0;
        self.settle = if switched { Self::SWITCH_SETTLE } else { Self::SETTLE };
        self.i = Some(i);
        self.t0 = t;
    }

    /// Advance; returns true when every point is done.
    fn update(&mut self, t: f64, tracker: &Tracker) -> bool {
        let Some(i) = self.i else { return false };
        if t - self.t0 < self.settle + Self::COLLECT {
            return false;
        }
        let (t0, t1) = (self.t0 + self.settle, self.t0 + self.settle + Self::COLLECT);
        let feats = tracker.features_between(t0, t1);
        let (mi, x, y) = self.points[i];
        if feats.len() >= 5 {
            self.model.add(&feats, [x, y], i as i64, 1.0);
            let ug: Vec<[f64; 2]> = tracker.ug_between(t0, t1).iter().map(|u| u.angles).collect();
            self.umodel.add(&ug, [x, y], i as i64, 1.0);
        } else if self.retried.insert((mi, x as i64, y as i64)) {
            self.points.push((mi, x, y)); // face not seen: try this dot again at the end
        }
        let frames = tracker.raw_between(t0, t1);
        use base64::Engine;
        self.raw.push(serde_json::json!({
            "monitor": mi, "target": [x, y],
            "frames": frames.iter().map(|f| f.0.as_slice().to_vec()).collect::<Vec<_>>(),
            "crops": frames.iter().map(|f| f.1.as_ref().map(|j| base64::engine::general_purpose::STANDARD.encode(j.as_slice()))).collect::<Vec<_>>(),
        }));
        if i + 1 >= self.points.len() {
            return true;
        }
        self.start_point(i + 1, t);
        false
    }
}

// ---------------------------------------------------------------------------------------------
struct Popup {
    action: Action,
    x0: f64,
    y0: f64,
    dwell_accept: f64,
    dwell_dismiss: f64,
    created: f64,
    result: Option<bool>, // Some(true) = accepted
}

impl Popup {
    const W: f32 = 600.0;
    const H: f32 = 270.0;
    const ACCEPT_DWELL: f64 = 1.0;
    const DISMISS_DWELL: f64 = 1.3;
    const TIMEOUT: f64 = 30.0;

    fn new(action: Action, mon: &Monitor) -> Self {
        Self {
            action,
            x0: mon.right() as f64 - Self::W as f64 - 24.0,
            y0: mon.top as f64 + 48.0,
            dwell_accept: 0.0,
            dwell_dismiss: 0.0,
            created: now(),
            result: None,
        }
    }

    fn need(gestures: bool) -> f64 {
        if gestures {
            Self::ACCEPT_DWELL
        } else {
            1.6
        }
    }

    /// `confirm`: a confirming hand pose (pinch or fist) is held right now.
    fn update(&mut self, gaze: Option<(f64, f64)>, confirm: bool, gestures: bool, dt: f64) {
        if self.result.is_some() {
            return;
        }
        let m = 90.0; // gaze is ~2-3 cm accurate: be generous around the popup
        let zone = gaze.and_then(|(gx, gy)| {
            let inside = gx >= self.x0 - m && gx <= self.x0 + Self::W as f64 + m && gy >= self.y0 - m && gy <= self.y0 + Self::H as f64 + m;
            inside.then_some(gx < self.x0 + Self::W as f64 / 2.0)
        });
        let step = |d: &mut f64, on: bool| *d = if on { *d + dt } else { (*d - 2.0 * dt).max(0.0) };
        step(&mut self.dwell_accept, zone == Some(true));
        step(&mut self.dwell_dismiss, zone == Some(false));
        let armed = self.dwell_accept >= Self::need(gestures);
        if armed && (confirm || !gestures) {
            self.result = Some(true);
        } else if self.dwell_dismiss >= Self::DISMISS_DWELL || now() - self.created > Self::TIMEOUT {
            self.result = Some(false);
        }
    }
}

// ---------------------------------------------------------------------------------------------
#[derive(Default)]
struct VoiceOverlay {
    visible: bool,
    pos: (f64, f64),
    head: String,
    text: String,
    target: Arc<Mutex<String>>,
    hide_at: Option<f64>,
}

impl VoiceOverlay {
    const W: f32 = 680.0;
    const H: f32 = 130.0;

    fn show(&mut self, mon: &Monitor) {
        self.visible = true;
        self.pos = (mon.left as f64 + (mon.width as f64 - Self::W as f64) / 2.0, mon.bottom() as f64 - 190.0);
        self.head = "🎙 Listening → window under your gaze".into();
        self.text = "…".into();
        *self.target.lock() = String::new();
        self.hide_at = None;
    }

    fn event(&mut self, e: dictation::Event) {
        let target = self.target.lock().clone();
        let head = |s: &str| if target.is_empty() { s.to_string() } else { format!("{s} → {target}") };
        let t = now();
        match e {
            dictation::Event::Status(s) => {
                if !s.is_empty() {
                    self.head = s;
                }
            }
            dictation::Event::Listening => self.head = head("🎙 Listening"),
            dictation::Event::Text(s) => self.text = s,
            dictation::Event::Sent(s) => {
                self.head = head("✅ Sent");
                if !s.is_empty() {
                    self.text = s;
                }
                self.hide_at = Some(t + 1.5);
            }
            dictation::Event::Done(s) => {
                self.head = head("✅ Typed");
                if !s.is_empty() {
                    self.text = s;
                }
                self.hide_at = Some(t + 1.5);
            }
            dictation::Event::Cancelled => {
                self.head = "↩ Cancelled (erased)".into();
                self.hide_at = Some(t + 1.5);
            }
            dictation::Event::Error(s) => {
                self.visible = true;
                self.head = "⚠ Voice typing error".into();
                self.text = s;
                self.hide_at = Some(t + 5.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
pub struct PrecogApp {
    saved: Settings,
    cfg: Settings,
    monitors: Vec<Monitor>,
    tracker: Tracker,
    model: Option<GazeModel>,
    /// UniGaze angles -> screen; fused with the landmark model (see `fused`).
    umodel: Option<GazeModel>,
    filter: FixationFilter,
    gaze: Option<(f64, f64)>,
    face: bool,
    agent: Agent,
    popup: Option<Popup>,
    dictation: Dictation,
    voice: VoiceOverlay,
    voice_starting: Arc<AtomicBool>,
    voice_cooldown: f64,
    pose_db: Debounce,
    prev_pose: Option<Pose>,
    pose: Option<Pose>,
    n_clicks: usize,
    calib: Option<Calibration>,
    want_calibration: bool,
    tray: Option<Tray>,
    tray_status: String,
    show_settings: bool,
    smudge: Smudge,
    corrector: ClickCorrector<GazeInput>,
    /// Gaze error measured at each click, before learning from it: the real-world accuracy.
    click_errs: std::collections::VecDeque<f64>,
    last: f64,
    custom_model: String,
}

pub struct Options {
    pub camera: u32,
    pub calibrate: bool,
    pub no_tray: bool,
    pub overrides: Vec<(&'static str, bool)>,
}

impl PrecogApp {
    pub fn new(cc: &eframe::CreationContext, opts: Options) -> Self {
        let saved = Settings::load();
        let mut cfg = saved.clone();
        for (k, v) in &opts.overrides {
            tray::set_bool(&mut cfg, k, *v); // CLI flags: this session only
        }
        let monitors = monitors::list();
        let tracker = Tracker::start(opts.camera, paths::onnx_dir());
        let mut model = None;
        let mut umodel = None;
        let path = paths::gaze_model();
        if path.exists() && !opts.calibrate {
            umodel = GazeModel::load(&paths::gaze_model_unigaze()).ok();
            match GazeModel::load(&path) {
                Ok(m) if m.version != FEATURE_VERSION => println!("gaze features changed since last calibration: recalibrating"),
                Ok(m) if m.layout == monitors.iter().map(|m| m.layout()).collect::<Vec<_>>() => model = Some(m),
                Ok(_) => println!("screen layout changed since last calibration: recalibrating"),
                Err(e) => println!("could not load gaze model ({e:#}): recalibrating"),
            }
        }
        let tray = if opts.no_tray {
            None
        } else {
            Tray::new(&cfg).map_err(|e| println!("tray icon unavailable: {e:#}")).ok()
        };
        let want_calibration = model.is_none();
        let mut app = Self {
            agent: Agent::new(monitors.clone()),
            umodel: if model.is_some() { umodel } else { None },
            saved,
            custom_model: cfg.model.clone(),
            cfg,
            monitors,
            tracker,
            model,
            filter: FixationFilter::new(100.0),
            gaze: None,
            face: false,
            popup: None,
            dictation: Dictation::new(),
            voice: VoiceOverlay::default(),
            voice_starting: Arc::new(AtomicBool::new(false)),
            voice_cooldown: 0.0,
            pose_db: Debounce::new(0.3),
            prev_pose: None,
            pose: None,
            n_clicks: 0,
            calib: None,
            want_calibration,
            tray,
            tray_status: String::new(),
            show_settings: false,
            smudge: Smudge::new(),
            corrector: ClickCorrector::new(),
            click_errs: Default::default(),
            last: now(),
        };
        app.apply();
        if let Some(m) = &app.model {
            app.filter = FixationFilter::new(m.cv_error() * 0.5);
        }
        if app.cfg.gesture && app.cfg.voice {
            app.dictation.preload();
        }
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        app
    }

    fn apply(&mut self) {
        let c = &self.cfg;
        self.agent.paused = !c.agent;
        self.agent.dwell_s = c.dwell;
        self.agent.min_interval = c.interval;
        self.agent.cfg = c.clone();
        self.tracker.use_hands.store(c.gesture, Ordering::Relaxed);
        self.tracker.mirrored.store(c.mirror, Ordering::Relaxed);
        self.dictation.configure(&c.stt_model, &c.stt_language);
    }

    /// Persist a settings change made in the UI or tray.
    fn commit(&mut self, new: Settings) {
        let old = std::mem::replace(&mut self.cfg, new);
        if old == self.cfg {
            return;
        }
        self.saved = self.cfg.clone();
        if let Err(e) = self.saved.save() {
            println!("could not save settings: {e:#}");
        }
        self.apply();
        if self.cfg.stt_model != old.stt_model || (self.cfg.voice && !old.voice) || (self.cfg.gesture && !old.gesture) {
            self.dictation.preload();
        }
        if let Some(t) = &self.tray {
            t.sync(&self.cfg);
        }
    }

    // ---- per-frame logic ---------------------------------------------------------------------
    fn step(&mut self, ctx: &egui::Context) {
        let t = now();
        let dt = t - self.last;
        self.last = t;

        if let Some(e) = self.tracker.shared.lock().error.take() {
            eprintln!("{e}");
            ctx.send_viewport_cmd(ViewportCommand::Close);
            return;
        }
        let f = self.tracker.latest();
        self.face = f.as_ref().is_some_and(|f| f.features.is_some() && t - f.t < 0.3);
        match (&f, &self.model) {
            (Some(fr), Some(m)) if self.face && self.calib.is_none() => {
                let ug = self.tracker.ug_latest().filter(|u| t - u.t < 1.0).and_then(|u| Some((u.angles, u.land?)));
                let input = GazeInput { land: fr.features.unwrap(), ug };
                if let Some((x, y)) = fused(m, self.umodel.as_ref(), &input) {
                    let c = self.corrector.correction([x, y], t);
                    let (x, y) = monitors::clamp(&self.monitors, x + c[0], y + c[1]);
                    let p = self.filter.update([x, y], t);
                    self.gaze = Some((p[0], p[1]));
                }
            }
            _ if !self.face => {
                self.gaze = None;
                self.filter.reset();
            }
            _ => {}
        }
        self.smudge.set_noise(self.filter.jitter(), dt);
        self.learn_from_clicks();
        self.handle_tray(ctx);

        if self.want_calibration && self.calib.is_none() {
            self.want_calibration = false;
            self.calib = Some(Calibration::new(&self.monitors));
        }
        self.tracker.record.store(self.calib.is_some(), Ordering::Relaxed);
        if let Some(c) = &mut self.calib {
            if c.update(t, &self.tracker) {
                self.finish_calibration(false);
            }
        }

        self.gestures(t, f.as_ref().and_then(|f| (t - f.t < 0.3).then_some(f.pose)).flatten());

        if self.calib.is_none() {
            if let Some(p) = &mut self.popup {
                let confirm = matches!(self.pose, Some(Pose::Pinch | Pose::Fist));
                p.update(self.gaze, confirm, self.cfg.gesture, dt);
                if let Some(ok) = p.result {
                    let a = self.popup.take().unwrap().action;
                    println!("[{}] {} — {}", if ok { "accept" } else { "dismiss" }, a.title, a.describe());
                    self.agent.last_call = now(); // cool down after a decision
                    if ok {
                        std::thread::spawn(move || {
                            if let Err(e) = actions::execute(&a) {
                                println!("action failed: {e:#}");
                            }
                        });
                    }
                }
            } else if self.dictation.state() == dictation::State::Idle && !self.voice_starting.load(Ordering::Relaxed) {
                if let Some((x, y)) = self.gaze {
                    self.agent.update(t, x, y);
                }
                if let Ok(a) = self.agent.suggestions.try_recv() {
                    let (x, y) = self.gaze.unwrap_or((self.monitors[0].left as f64, self.monitors[0].top as f64));
                    self.popup = Some(Popup::new(a, &monitors::at(&self.monitors, x, y)));
                }
            }
        }

        while let Ok(e) = self.dictation.events.try_recv() {
            self.voice.event(e);
        }
        if self.voice.hide_at.is_some_and(|h| t > h) {
            self.voice.visible = false;
            self.voice.hide_at = None;
        }

        let status = if self.calib.is_some() {
            "calibrating…".to_string()
        } else if self.dictation.state() != dictation::State::Idle {
            format!("🎙 {:?}", self.dictation.state()).to_lowercase()
        } else {
            format!("{} {}", if self.face { "●" } else { "○" }, if self.agent.paused { "paused".into() } else { self.agent.status.lock().clone() })
        };
        if status != self.tray_status {
            if let Some(tr) = &self.tray {
                tr.set_status(&format!("precog: {status}"));
            }
            self.tray_status = status;
        }
        ctx.request_repaint_after(Duration::from_millis(33));
    }

    fn handle_tray(&mut self, ctx: &egui::Context) {
        let Some(tr) = &self.tray else { return };
        for cmd in tr.poll() {
            match cmd {
                tray::Cmd::Toggle(k) => {
                    let mut c = self.cfg.clone();
                    tray::set_bool(&mut c, k, !tray::get_bool(&self.cfg, k));
                    self.commit(c);
                }
                tray::Cmd::Ask => self.agent.trigger(self.gaze),
                tray::Cmd::Calibrate => self.want_calibration = true,
                tray::Cmd::Settings => {
                    self.show_settings = true;
                    self.custom_model = self.cfg.model.clone();
                }
                tray::Cmd::Quit => ctx.send_viewport_cmd(ViewportCommand::Close),
            }
        }
    }

    fn finish_calibration(&mut self, abort: bool) {
        let Some(mut c) = self.calib.take() else { return };
        if abort || c.model.len() < 30 {
            println!("calibration cancelled/failed (is your face visible?)");
            return;
        }
        let raw_path = paths::data_dir().join("calibrations").join(format!(
            "{}.json",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
        ));
        let _ = std::fs::create_dir_all(raw_path.parent().unwrap());
        let _ = std::fs::write(&raw_path, serde_json::to_vec(&serde_json::json!({ "layout": c.model.layout, "points": c.raw })).unwrap_or_default());
        match c.model.fit() {
            Ok(err) => {
                let _ = c.model.save(&paths::gaze_model());
                self.filter = FixationFilter::new(err * 0.5);
                println!(
                    "calibrated on {} screen(s): {} samples, cross-validated median error {err:.0}px",
                    self.monitors.len(),
                    c.model.len()
                );
                self.model = Some(c.model);
                self.umodel = match c.umodel.fit() {
                    Ok(e) => {
                        println!("UniGaze: {} samples, cross-validated median error {e:.0}px", c.umodel.len());
                        let _ = c.umodel.save(&paths::gaze_model_unigaze());
                        Some(c.umodel)
                    }
                    Err(e) => {
                        println!("UniGaze not calibrated ({e:#}): landmarks only");
                        None
                    }
                };
                self.n_clicks = 0;
                self.corrector = ClickCorrector::new();
                self.click_errs.clear();
            }
            Err(e) => println!("calibration failed: {e:#}"),
        }
    }

    /// Continuous recalibration: people look at what they click ~100-300 ms before the click. Each click is
    /// scored first (real-world accuracy), skipped if the user clearly wasn't looking, then learned by both models
    /// (refit) and by the local corrector. Silent, no UI.
    fn learn_from_clicks(&mut self) {
        let mut learned = false;
        while let Ok((t, x, y)) = self.tracker.clicks.try_recv() {
            if !self.cfg.learn || self.calib.is_some() || self.popup.is_some() {
                continue;
            }
            let Some(m) = &mut self.model else { continue };
            let feats: Vec<[f64; NF]> = self.tracker.features_between(t - 0.35, t - 0.05);
            if feats.len() < 3 {
                continue;
            }
            let mut mean = [0.0; NF];
            for f in &feats {
                for i in 0..NF {
                    mean[i] += f[i] / feats.len() as f64;
                }
            }
            // UniGaze runs at ~10 Hz: take its results from a slightly wider window before the click
            let ugs = self.tracker.ug_between(t - 0.6, t - 0.02);
            let ug = ugs.iter().rev().find_map(|u| Some((u.angles, u.land?)));
            let input = GazeInput { land: mean, ug };
            if let Some((px, py)) = fused(m, self.umodel.as_ref(), &input) {
                let c = self.corrector.correction([px, py], t);
                let err = ((px + c[0] - x).powi(2) + (py + c[1] - y).powi(2)).sqrt();
                let med = median(&self.click_errs);
                if self.click_errs.len() >= 5 && err > (3.0 * med).max(700.0) {
                    continue; // looking elsewhere while clicking: not a valid label
                }
                self.click_errs.push_back(err);
                while self.click_errs.len() > 30 {
                    self.click_errs.pop_front();
                }
            }
            m.add(&feats, [x, y], CLICK_GROUP + self.n_clicks as i64, 1.0);
            if let Some(um) = &mut self.umodel {
                let angles: Vec<[f64; 2]> = ugs.iter().map(|u| u.angles).collect();
                um.add(&angles, [x, y], CLICK_GROUP + self.n_clicks as i64, 1.0);
            }
            self.corrector.add(t, [x, y], input);
            self.n_clicks += 1;
            learned = true;
        }
        if learned {
            if let Some(m) = &mut self.model {
                let _ = m.refit();
            }
            if let Some(um) = &mut self.umodel {
                let _ = um.refit();
            }
            if let Some(m) = &self.model {
                let um = self.umodel.as_ref();
                self.corrector.update_residuals(|inp| fused(m, um, inp));
                if self.n_clicks % 10 == 0 {
                    let _ = m.save(&paths::gaze_model());
                    if let Some(um) = um {
                        let _ = um.save(&paths::gaze_model_unigaze());
                    }
                }
            }
        }
    }

    // ---- gestures (no LLM) ---------------------------------------------------------------------
    fn gestures(&mut self, t: f64, raw_pose: Option<Pose>) {
        let swipes: Vec<Swipe> = self.tracker.swipes.try_iter().map(|s| s.1).collect();
        if !self.cfg.gesture || self.calib.is_some() {
            self.pose = None;
            return;
        }
        let pose = self.pose_db.update(t, raw_pose);
        let rising = if pose != self.prev_pose { pose } else { None };
        self.prev_pose = pose;
        self.pose = pose;
        let left = swipes.contains(&Swipe::Left);

        if let Some(p) = &mut self.popup {
            if left {
                p.result = Some(false);
            }
            return;
        }
        if !self.cfg.voice {
            return;
        }
        match self.dictation.state() {
            dictation::State::Idle if !self.voice_starting.load(Ordering::Relaxed) => {
                if rising == Some(Pose::Open) && t > self.voice_cooldown {
                    if let Some((x, y)) = self.gaze {
                        self.start_voice(x, y);
                    }
                }
            }
            dictation::State::Listening => {
                if left {
                    self.dictation.cancel();
                    self.voice_cooldown = t + 1.5;
                } else if rising == Some(Pose::Fist) {
                    self.dictation.send();
                    self.voice_cooldown = t + 1.5;
                } else if pose.is_none() && self.dictation.silent_for() > 5.0 {
                    self.dictation.stop();
                    self.voice_cooldown = t + 1.0;
                }
            }
            _ => {}
        }
    }

    fn start_voice(&mut self, x: f64, y: f64) {
        self.voice_starting.store(true, Ordering::Relaxed);
        self.voice.show(&monitors::at(&self.monitors, x, y));
        let (starting, target, click) = (self.voice_starting.clone(), self.voice.target.clone(), self.cfg.click_focus);
        let start = self.dictation.starter();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100)); // overlay first: it must not take focus back
            *target.lock() = actions::focus_at(x, y, click);
            start();
            std::thread::sleep(Duration::from_millis(150)); // let the worker switch to Listening
            starting.store(false, Ordering::Relaxed);
        });
    }

    // ---- drawing -------------------------------------------------------------------------------
    fn hud(&mut self, ui: &mut egui::Ui) {
        let mut parts = vec![format!("{:4.1}fps", self.tracker.shared.lock().fps), if self.face { "face ✓" } else { "face ✗" }.into()];
        if self.cfg.gesture {
            parts.push(Pose::icon(self.pose).into());
        }
        let ug = self.tracker.ug_rate();
        if ug > 0.0 {
            parts.push(format!("unigaze {ug:.0}Hz{}", if self.umodel.is_some() { "" } else { " (uncalibrated)" }));
        }
        if !self.click_errs.is_empty() {
            parts.push(format!("click err~{:.0}px ({} clicks)", median(&self.click_errs), self.n_clicks));
        } else if let Some(m) = &self.model {
            parts.push(format!("calib err~{:.0}px", m.cv_error()));
        }
        if self.dictation.state() != dictation::State::Idle {
            parts.push(format!("🎙 {:?}", self.dictation.state()).to_lowercase());
        }
        parts.push(format!("agent: {}", if self.agent.paused { "paused".into() } else { self.agent.status.lock().clone() }));
        let r = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(17, 17, 17, 235))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.label(egui::RichText::new(format!("precog · {}", parts.join(" · "))).monospace().color(Color32::from_gray(220)));
                ui.label(egui::RichText::new("click: ask now · right-click: pause suggestions").small().color(Color32::GRAY));
            });
        let resp = ui.interact(r.response.rect, egui::Id::new("hud"), egui::Sense::click());
        if resp.clicked() {
            self.agent.trigger(self.gaze);
        }
        if resp.secondary_clicked() {
            let mut c = self.cfg.clone();
            c.agent = !c.agent;
            self.commit(c);
        }
    }

    fn gaze_cloud(&mut self, ctx: &egui::Context) {
        let Some((x, y)) = self.gaze else { return };
        if !self.cfg.dot || self.calib.is_some() {
            return;
        }
        let s = cloud::SIZE;
        // Keep the window inside the monitor (macOS pushes windows that stick out above the screen back down,
        // which made the smudge stop short of the top) and draw the smudge offset inside it instead.
        let m = monitors::at(&self.monitors, x, y);
        let wx = (x as f32 - s / 2.0).clamp(m.left as f32, (m.right() as f32 - s).max(m.left as f32));
        let wy = (y as f32 - s / 2.0).clamp(m.top as f32, (m.bottom() as f32 - s).max(m.top as f32));
        let local = Pos2::new(x as f32 - wx, y as f32 - wy);
        let smudge = &mut self.smudge;
        // click-through, so it never needs to dodge the cursor
        ctx.show_viewport_immediate(
            ViewportId::from_hash_of("gaze"),
            overlay("precog gaze", [s, s], [wx, wy]).with_mouse_passthrough(true),
            |ui, _| {
                let ctx = ui.ctx().clone();
                egui::CentralPanel::no_frame().show(ui, |ui| smudge.paint(&ctx, ui.painter(), local, now()));
            },
        );
    }

    fn calibration_windows(&mut self, ctx: &egui::Context) {
        let Some(c) = &self.calib else { return };
        let (cur, t0, settle) = (c.i.map(|i| (i, c.points[i])), c.t0, c.settle);
        let total = c.points.len();
        let face = self.face;
        let mut start = false;
        let mut abort = false;
        for (mi, m) in self.monitors.iter().enumerate() {
            let b = ViewportBuilder::default()
                .with_title("precog calibration")
                .with_decorations(false)
                .with_always_on_top()
                .with_position([m.left as f32, m.top as f32])
                .with_inner_size([m.width as f32, m.height as f32])
                .with_taskbar(false);
            ctx.show_viewport_immediate(ViewportId::from_hash_of(("calib", mi)), b, |ui, _| {
                egui::CentralPanel::no_frame().frame(Frame::new().fill(Color32::BLACK)).show(ui, |ui| {
                    let p = ui.painter();
                    let size = ui.max_rect().size();
                    match cur {
                        None => {
                            p.text(
                                Pos2::new(size.x / 2.0, size.y / 2.0),
                                Align2::CENTER_CENTER,
                                format!(
                                    "Calibration · {} screen(s)\n\nKeep your eyes on each dot, and while you do, gently move your head a little\n(left/right, up/down, closer/further): it teaches precog to handle head movement.\nSit the way you usually work.\n\nCLICK anywhere to start · ESC to cancel",
                                    self.monitors.len()
                                ),
                                FontId::proportional(26.0),
                                Color32::WHITE,
                            );
                        }
                        Some((i, (pm, x, y))) if pm == mi => {
                            let el = now() - t0;
                            let r = if el < settle { (34.0 * (1.0 - el / settle)).max(6.0) } else { 6.0 } as f32;
                            let c = Pos2::new((x - m.left as f64) as f32, (y - m.top as f64) as f32);
                            p.circle_filled(c, r, RED);
                            p.circle_filled(c, 2.0, Color32::WHITE);
                            p.text(Pos2::new(size.x / 2.0, 30.0), Align2::CENTER_CENTER, format!("{}/{total} · eyes on the dot, move your head gently", i + 1), FontId::proportional(14.0), Color32::GRAY);
                            if !face {
                                p.text(Pos2::new(size.x / 2.0, 60.0), Align2::CENTER_CENTER, "⚠ face not detected: face the camera, add light", FontId::proportional(18.0), Color32::from_rgb(255, 180, 0));
                            }
                        }
                        _ => {}
                    }
                });
                ui.input(|i| {
                    start |= i.pointer.any_click();
                    abort |= i.key_pressed(egui::Key::Escape);
                });
            });
        }
        if abort {
            self.finish_calibration(true);
        } else if start && cur.is_none() {
            if let Some(c) = &mut self.calib {
                c.start_point(0, now());
            }
        }
    }

    fn popup_window(&mut self, ctx: &egui::Context) {
        let gestures = self.cfg.gesture;
        let Some(p) = &mut self.popup else { return };
        let need = Popup::need(gestures);
        ctx.show_viewport_immediate(
            ViewportId::from_hash_of("popup"),
            overlay("precog suggestion", [Popup::W, Popup::H], [p.x0 as f32, p.y0 as f32]),
            |ui, _| {
                egui::CentralPanel::no_frame().frame(Frame::new().fill(BG).inner_margin(16)).show(ui, |ui| {
                    ui.label(egui::RichText::new(format!("🔮 {}", p.action.title)).size(18.0).strong().color(Color32::WHITE));
                    ui.label(egui::RichText::new(&p.action.reason).color(Color32::LIGHT_GRAY));
                    Frame::new().fill(CODE_BG).inner_margin(8).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(p.action.describe()).monospace().color(GREEN)).wrap());
                    });
                    ui.add_space(8.0);
                    ui.columns(2, |cols| {
                        let zones = [
                            (true, Color32::from_rgb(31, 111, 63), p.dwell_accept / need, if gestures { "✓ ACCEPT\nlook + pinch 🤏 or ✊" } else { "✓ ACCEPT\nlook here" }),
                            (false, Color32::from_rgb(90, 31, 31), p.dwell_dismiss / Popup::DISMISS_DWELL, if gestures { "✗ DISMISS\nlook here · or swipe 👈" } else { "✗ DISMISS\nlook here" }),
                        ];
                        for (col, (accept, color, frac, label)) in cols.iter_mut().zip(zones) {
                            let (rect, resp) = col.allocate_exact_size(egui::vec2(col.available_width(), 100.0), egui::Sense::click());
                            let painter = col.painter();
                            painter.rect_filled(rect, 6.0, color);
                            let mut bar = rect;
                            bar.set_top(rect.bottom() - 8.0);
                            bar.set_width(rect.width() * frac.min(1.0) as f32);
                            painter.rect_filled(bar, 0.0, Color32::WHITE);
                            painter.text(rect.center(), Align2::CENTER_CENTER, label, FontId::proportional(17.0), Color32::WHITE);
                            if resp.clicked() {
                                p.result = Some(accept);
                            }
                        }
                    });
                });
            },
        );
    }

    fn voice_window(&mut self, ctx: &egui::Context) {
        if !self.voice.visible {
            return;
        }
        let v = &self.voice;
        ctx.show_viewport_immediate(
            ViewportId::from_hash_of("voice"),
            overlay("precog voice", [VoiceOverlay::W, VoiceOverlay::H], [v.pos.0 as f32, v.pos.1 as f32]),
            |ui, _| {
                egui::CentralPanel::no_frame().frame(Frame::new().fill(BG).inner_margin(12)).show(ui, |ui| {
                    ui.label(egui::RichText::new(&v.head).size(15.0).strong().color(Color32::WHITE));
                    Frame::new().fill(CODE_BG).inner_margin(6).show(ui, |ui| {
                        let tail: String = v.text.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
                        ui.add(egui::Label::new(egui::RichText::new(tail).monospace().color(GREEN)).wrap());
                    });
                    ui.label(egui::RichText::new("✊ close hand: send ⏎  ·  👈 swipe left: cancel  ·  lower hand + pause: stop").small().color(Color32::GRAY));
                });
            },
        );
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }
        let mut c = self.cfg.clone();
        let mut close = false;
        let mut recalibrate = false;
        let custom = &mut self.custom_model;
        ctx.show_viewport_immediate(
            ViewportId::from_hash_of("settings"),
            ViewportBuilder::default().with_title("precog settings").with_inner_size([520.0, 680.0]).with_resizable(true),
            |ui, _| {
                egui::CentralPanel::default_margins().frame(Frame::central_panel(ui.style()).fill(BG)).show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| settings_form(ui, &mut c, custom, &mut recalibrate));
                });
                close = ui.input(|i| i.viewport().close_requested());
            },
        );
        if recalibrate {
            self.want_calibration = true;
        }
        if close {
            self.show_settings = false;
        }
        self.commit(c);
    }
}

/// What the gaze estimate is computed from (also stored per click, to recompute residuals after refits).
#[derive(Clone)]
struct GazeInput {
    /// Landmark features now.
    land: [f64; NF],
    /// Latest UniGaze result: (angles, landmark features of the frame it was computed on).
    ug: Option<([f64; 2], [f64; NF])>,
}

/// UniGaze gives the accurate absolute position (~10 Hz); the landmark model adds how far the gaze moved since
/// that frame, so the estimate keeps the 30 fps responsiveness (saccades) of the landmarks.
fn fused(land: &GazeModel, uni: Option<&GazeModel>, inp: &GazeInput) -> Option<(f64, f64)> {
    let (lx, ly) = land.predict(&inp.land)?;
    if let (Some(um), Some((angles, land_then))) = (uni.filter(|u| u.ready()), &inp.ug) {
        if let (Some((ux, uy)), Some((tx, ty))) = (um.predict(angles), land.predict(land_then)) {
            return Some((ux + lx - tx, uy + ly - ty));
        }
    }
    Some((lx, ly))
}

fn median(v: &std::collections::VecDeque<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s: Vec<f64> = v.iter().copied().collect();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn overlay(title: &str, size: [f32; 2], pos: [f32; 2]) -> ViewportBuilder {
    ViewportBuilder::default()
        .with_title(title)
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top()
        .with_inner_size(size)
        .with_position(pos)
        .with_taskbar(false)
        .with_active(false)
        .with_has_shadow(false)
        .with_resizable(false)
}

fn settings_form(ui: &mut egui::Ui, c: &mut Settings, custom_model: &mut String, recalibrate: &mut bool) {
    ui.heading("General");
    ui.checkbox(&mut c.agent, "Suggestions (LLM looks at the screen when your gaze settles)");
    ui.checkbox(&mut c.dot, "Show gaze cloud");
    ui.checkbox(&mut c.learn, "Learn from my clicks (improves gaze accuracy over time)");
    ui.add(egui::Slider::new(&mut c.dwell, 1.0..=4.0).text("fixation before asking (s)"));
    ui.add(egui::Slider::new(&mut c.interval, 5.0..=120.0).logarithmic(true).text("min time between asks (s)"));
    if ui.button("Recalibrate gaze…").clicked() {
        *recalibrate = true;
    }

    ui.separator();
    ui.heading("Gestures & voice");
    ui.checkbox(&mut c.gesture, "Hand gestures (🤏/✊ accept · 👈 swipe cancel)");
    ui.add_enabled(c.gesture, egui::Checkbox::new(&mut c.voice, "✋ Open hand → voice typing into the window I look at · ✊ send"));
    ui.checkbox(&mut c.click_focus, "Click where I look to focus the field (gaze ~2-3 cm accurate)");
    ui.checkbox(&mut c.mirror, "Invert swipe direction");
    egui::ComboBox::from_label("speech model")
        .selected_text(&c.stt_model)
        .show_ui(ui, |ui| {
            for m in ["tiny", "base", "small", "large-v3-turbo"] {
                ui.selectable_value(&mut c.stt_model, m.to_string(), m);
            }
        });
    let langs = [("", "Auto-detect"), ("en", "English"), ("fr", "Français"), ("de", "Deutsch"), ("es", "Español")];
    egui::ComboBox::from_label("language")
        .selected_text(langs.iter().find(|l| l.0 == c.stt_language).map_or(c.stt_language.as_str(), |l| l.1))
        .show_ui(ui, |ui| {
            for (code, label) in langs {
                ui.selectable_value(&mut c.stt_language, code.to_string(), label);
            }
        });

    ui.separator();
    ui.heading("AI provider");
    let before = c.provider.clone();
    egui::ComboBox::from_label("provider").selected_text(llm::provider(&c.provider).label).show_ui(ui, |ui| {
        for p in llm::PROVIDERS {
            ui.selectable_value(&mut c.provider, p.id.to_string(), p.label);
        }
    });
    if c.provider != before {
        c.model.clear(); // new provider -> its default model
        custom_model.clear();
    }
    let (p, model) = llm::resolve(c);
    ui.label(egui::RichText::new(format!("using: {} · {}", p.label, if model.is_empty() { "(set a model)" } else { &model })).small());
    if !p.presets.is_empty() {
        egui::ComboBox::from_label("model preset").selected_text(&model).show_ui(ui, |ui| {
            for m in p.presets {
                if ui.selectable_label(model == *m, *m).clicked() {
                    c.model = m.to_string();
                    *custom_model = c.model.clone();
                }
            }
        });
    }
    ui.horizontal(|ui| {
        ui.label("model");
        if ui.text_edit_singleline(custom_model).lost_focus() {
            c.model = custom_model.trim().to_string();
        }
    });
    if p.id == "openai" || c.provider == "openai" {
        egui::ComboBox::from_label("endpoint preset").selected_text(&c.base_url).show_ui(ui, |ui| {
            for (label, url) in llm::ENDPOINTS {
                ui.selectable_value(&mut c.base_url, url.to_string(), format!("{label}  {url}"));
            }
        });
        ui.horizontal(|ui| {
            ui.label("base URL");
            ui.text_edit_singleline(&mut c.base_url);
        });
    }
    ui.horizontal(|ui| {
        ui.label("API key");
        ui.add(egui::TextEdit::singleline(&mut c.api_key).password(true).hint_text("empty = env variable"));
    });
    ui.horizontal(|ui| {
        ui.label("API key env var");
        ui.add(egui::TextEdit::singleline(&mut c.api_key_env).hint_text(if p.key_env.is_empty() { "-" } else { p.key_env }));
    });
    ui.add_space(8.0);
    ui.label(egui::RichText::new(format!("Saved to {} (mode 600)", crate::settings::path().display())).small().color(Color32::GRAY));
}

impl Drop for PrecogApp {
    fn drop(&mut self) {
        if let Some(m) = &self.model {
            let _ = m.save(&paths::gaze_model()); // keep what the clicks taught
        }
        if let Some(um) = &self.umodel {
            let _ = um.save(&paths::gaze_model_unigaze());
        }
        if self.n_clicks > 0 {
            println!(
                "session: learned from {} clicks, median gaze error at the last {} clicks: {:.0}px",
                self.n_clicks,
                self.click_errs.len(),
                median(&self.click_errs)
            );
        }
    }
}

impl eframe::App for PrecogApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.step(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.hud(ui);
        self.gaze_cloud(&ctx);
        self.calibration_windows(&ctx);
        self.popup_window(&ctx);
        self.voice_window(&ctx);
        self.settings_window(&ctx);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0; 4] // transparent: overlays paint their own backgrounds
    }
}
