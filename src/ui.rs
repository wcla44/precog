//! egui app: HUD (root window), gaze ring, multi-monitor calibration, suggestion popup, voice overlay,
//! settings window. Core logic runs in `logic()` every frame; drawing in `ui()`.

use crate::actions::{self, Action};
use crate::calib::{Calibration, Outcome};
use crate::capture;
use crate::agent::Agent;
use crate::cloud::{self, Smudge};
use crate::dictation::{self, Dictation};
use crate::gaze::{self, ClickCorrector, GazeKalman, GazeModel, CLICK_GROUP, FEATURE_VERSION, NF};
use crate::halo::{Halo, LeftHand};
use crate::gestures::{Debounce, Pose, Swipe};
use crate::monitors::{self, Monitor};
use crate::settings::Settings;
use crate::tracker::{Frame as TrackerFrame, Tracker};
use crate::tray::{self, Tray};
use crate::{llm, now, paths};
use egui::{Align2, Color32, CornerRadius, FontId, Frame, Margin, Pos2, Stroke, ViewportBuilder, ViewportCommand, ViewportId};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const BG: Color32 = Color32::from_rgb(28, 28, 30);
const CODE_BG: Color32 = Color32::from_rgb(13, 17, 23);
const GREEN: Color32 = Color32::from_rgb(126, 231, 135);
const RED: Color32 = Color32::from_rgb(255, 59, 48);
/// The open hand has to be held this long to start voice typing.
const VOICE_HOLD: f64 = 0.5;

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
        self.head = "Listening -> window under your gaze".into();
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
            dictation::Event::Listening => self.head = head("Listening"),
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
                self.head = "Voice typing error".into();
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
    filter: GazeKalman,
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
    halo: Halo,
    left: Option<LeftHand>,
    /// Halo capture running in the background (the halo is hidden meanwhile, so it isn't in the screenshot).
    capturing: Arc<AtomicBool>,
    captures: (crossbeam_channel::Sender<String>, crossbeam_channel::Receiver<String>),
    flash_until: f64,
    /// Short message in the HUD, until the given time.
    notice: Option<(String, f64)>,
    /// Posture (face position in the camera image) the gaze is calibrated for, since when the user sits away from
    /// it, and clicks learned since.
    posture_ref: Option<[f64; 3]>,
    moved_since: Option<f64>,
    nudges_since_move: usize,
    last_debug: f64,
    /// X11: a compositor is running, so the halo can be Lucien's translucent smudge (else: a shaped red ring).
    /// Rechecked every 5 s.
    compositing: (bool, f64),
    /// New screens waiting for their quick calibration (their pose in the geometric model).
    pending_screens: Vec<Monitor>,
    /// Last check of the screen setup (plugged / unplugged / moved screens are followed live).
    screens_checked: f64,
    /// Self view texture during calibrations, and the preview frame it shows.
    preview_tex: Option<(egui::TextureHandle, u64)>,
    /// Keyboard state for the fallback shortcuts (Ctrl+Alt+H/C/V/X/Q), and the last combination seen.
    keys: Option<device_query::DeviceState>,
    last_combo: Option<device_query::Keycode>,
    /// Right hand open since (voice typing hold), and whether that hold already started dictation.
    open_since: Option<f64>,
    open_used: bool,
    /// Last time a hand was seen (the gesture panel stays a moment after).
    hand_seen: f64,
    /// Debug timings: worst logic and ui time (s), frames, since.
    timing: (f64, f64, u32, f64),
    /// The HUD is moved off screen during calibrations.
    hud_away: bool,
    /// Last pointer position over a calibration window (the laser).
    calib_pointer: Option<(f64, f64)>,
    /// The ring window is clipped to its shape (until then it stays hidden: it would show as a filled square).
    halo_shaped: bool,
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
        let mut new_screens = Vec::new();
        let path = paths::gaze_model();
        if path.exists() && !opts.calibrate {
            umodel = GazeModel::load(&paths::gaze_model_unigaze()).ok();
            let layout: Vec<_> = monitors.iter().map(|m| m.layout()).collect();
            match GazeModel::load(&path) {
                Ok(m) if m.version != FEATURE_VERSION => println!("gaze features changed since last calibration: recalibrating"),
                Ok(m) if m.layout == layout => model = Some(m),
                // other screens than at calibration: the geometric model adapts (only new screens need calibrating)
                Ok(mut m) => match m.adapt_to(&monitors) {
                    Ok(new) => {
                        println!("screens changed since the calibration: gaze model adapted ({} new screen(s) to calibrate)", new.len());
                        umodel = None; // the UniGaze model maps to pixels of the old layout
                        new_screens = new;
                        model = Some(m);
                    }
                    Err(e) => println!("screen layout changed since last calibration ({e:#}): recalibrating"),
                },
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
            filter: GazeKalman::new(100.0),
            gaze: None,
            face: false,
            popup: None,
            dictation: Dictation::new(),
            voice: VoiceOverlay::default(),
            voice_starting: Arc::new(AtomicBool::new(false)),
            voice_cooldown: 0.0,
            pose_db: Debounce::new(0.15),
            prev_pose: None,
            pose: None,
            n_clicks: 0,
            calib: None,
            want_calibration,
            tray,
            tray_status: String::new(),
            show_settings: false,
            smudge: Smudge::new(),
            halo: Halo::new(),
            left: None,
            capturing: Arc::new(AtomicBool::new(false)),
            captures: crossbeam_channel::unbounded(),
            flash_until: 0.0,
            notice: None,
            posture_ref: None,
            moved_since: None,
            nudges_since_move: 0,
            last_debug: 0.0,
            compositing: (true, -1e9),
            halo_shaped: false,
            calib_pointer: None,
            pending_screens: new_screens,
            screens_checked: 0.0,
            hud_away: false,
            timing: (0.0, 0.0, 0, 0.0),
            preview_tex: None,
            #[cfg(target_os = "macos")]
            keys: device_query::DeviceState::checked_new(), // needs Accessibility; no shortcuts without it
            #[cfg(not(target_os = "macos"))]
            keys: Some(device_query::DeviceState::new()),
            last_combo: None,
            open_since: None,
            open_used: false,
            hand_seen: -1e9,
            corrector: ClickCorrector::new(),
            click_errs: Default::default(),
            last: now(),
        };
        app.apply();
        if let Some(m) = &app.model {
            if let Some(f) = m.focal {
                crate::unigaze::set_focal(f); // the camera focal length the model was calibrated with
            }
            app.filter = GazeKalman::new(m.cv_error() * 0.5);
            app.posture_ref = m.calibrated_head();
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
        self.check_posture(t, f.as_ref());
        self.learn_from_clicks();
        self.handle_tray(ctx);

        if t - self.screens_checked > 2.0 {
            self.screens_checked = t;
            let t_list = now();
            let now_mons = monitors::list();
            if std::env::var_os("PRECOG_DEBUG").is_some() {
                println!("[timing] screen check {:.1} ms", (now() - t_list) * 1000.0);
            }
            let key = |v: &[Monitor]| v.iter().map(|m| (m.layout(), m.mm.map(|(a, b)| (a as i64, b as i64)), m.builtin)).collect::<Vec<_>>();
            if key(&now_mons) != key(&self.monitors) {
                self.screens_changed(ctx, now_mons);
            }
        }
        if !self.pending_screens.is_empty() && self.calib.is_none() && self.model.is_some() {
            let screens = std::mem::take(&mut self.pending_screens);
            self.say("New screen: quick calibration of where it is", 4.0);
            self.calib = Some(Calibration::for_screens(&self.monitors, &screens));
        }
        if self.want_calibration && self.calib.is_none() {
            self.want_calibration = false;
            self.calib = Some(Calibration::new(&self.monitors));
        }
        self.tracker.record.store(self.calib.is_some(), Ordering::Relaxed);
        if let Some(c) = &mut self.calib {
            if let Outcome::Finished = c.update(t, &self.tracker) {
                self.finish_calibration(false);
            }
        }

        let fresh = f.as_ref().filter(|f| t - f.t < 0.3);
        self.left = if self.cfg.gesture { fresh.and_then(|f| f.left) } else { None };
        if self.cfg.dot && self.calib.is_none() {
            if !self.halo.sized_by_hand && !self.halo.frozen {
                // until the left hand sizes it, the halo grows with the gaze noise (Lucien's smudge sizing)
                self.smudge.set_noise(self.filter.jitter(), dt);
                self.halo.radius = self.smudge.radius() as f64;
            }
            self.halo.follow(t, self.gaze, dt);
            if self.halo.left_hand(t, self.left, dt) {
                self.say("Halo frozen · right fist or pinch: copy what's under it · right swipe left: release", 6.0);
            }
        } else if self.halo.frozen {
            self.halo.release();
        }
        for _ in self.tracker.shakes.try_iter().collect::<Vec<_>>() {
            if self.cfg.dot && self.calib.is_none() {
                self.halo.reset();
                self.say("Halo reset (head shake)", 2.0);
            }
        }
        while let Ok(msg) = self.captures.1.try_recv() {
            self.flash_until = t + 0.6;
            self.say(&msg, 4.0);
        }
        if self.notice.as_ref().is_some_and(|n| t > n.1) {
            self.notice = None;
        }

        self.gestures(t, fresh.and_then(|f| f.pose));
        self.shortcuts(ctx);
        if self.left.is_some() || self.pose.is_some() {
            self.hand_seen = t;
        }
        self.debug_log(t, fresh);

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
            format!("voice {:?}", self.dictation.state()).to_lowercase()
        } else {
            format!("{} {}", if self.face { "●" } else { "○" }, if self.agent.paused { "paused".into() } else { self.agent.status.lock().clone() })
        };
        if status != self.tray_status {
            if let Some(tr) = &self.tray {
                tr.set_status(&format!("precog: {status}"));
            }
            self.tray_status = status;
        }
        if self.calib.as_ref().is_some_and(|c| c.animating()) {
            // the laser follows the mouse and the pursuit dot moves: redraw at the display rate (vsync)
            ctx.request_repaint();
        } else {
            // 60 Hz while the halo glides, so it moves smoothly; 30 Hz otherwise
            ctx.request_repaint_after(Duration::from_millis(if self.halo.moving() { 16 } else { 33 }));
        }
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
                tray::Cmd::QuickCalibrate => self.quick_calibration(),
                tray::Cmd::Settings => {
                    self.show_settings = true;
                    self.custom_model = self.cfg.model.clone();
                }
                tray::Cmd::Quit => ctx.send_viewport_cmd(ViewportCommand::Close),
            }
        }
    }

    /// The screen setup changed while running: follow it (geometric model), or recalibrate.
    fn screens_changed(&mut self, ctx: &egui::Context, mons: Vec<Monitor>) {
        println!("screens changed: {} -> {} screen(s)", self.monitors.len(), mons.len());
        if self.calib.take().is_some() {
            println!("calibration interrupted by the screen change");
        }
        self.monitors = mons.clone();
        self.agent.monitors = mons.clone();
        self.corrector = ClickCorrector::new(); // residuals in pixels of the old layout
        self.click_errs.clear();
        self.umodel = None;
        self.popup = None;
        self.halo.reset();
        let p = &self.monitors[0];
        ctx.send_viewport_cmd(ViewportCommand::InnerSize(egui::vec2(720.0, 52.0)));
        ctx.send_viewport_cmd(ViewportCommand::OuterPosition(Pos2::new(p.left as f32 + 12.0, p.bottom() as f32 - 70.0)));
        self.hud_away = false;
        match self.model.as_mut().map(|m| m.adapt_to(&mons)) {
            Some(Ok(new)) => {
                if let Some(m) = &self.model {
                    let _ = m.save(&paths::gaze_model());
                }
                if new.is_empty() {
                    self.say(&format!("Screens changed ({}): gaze adapted, no calibration needed", mons.len()), 5.0);
                } else {
                    self.pending_screens = new;
                }
            }
            _ => {
                self.model = None;
                self.want_calibration = true;
            }
        }
    }

    fn quick_calibration(&mut self) {
        if self.model.is_some() && self.calib.is_none() {
            self.calib = Some(Calibration::quick(&self.monitors));
        } else if self.model.is_none() {
            self.want_calibration = true;
        }
    }

    fn finish_calibration(&mut self, abort: bool) {
        let Some(mut c) = self.calib.take() else { return };
        if c.quick {
            if !abort {
                self.finish_quick(&c);
            }
            return;
        }
        // the recording is kept even when cancelled: it is what tells why a calibration didn't work (--replay)
        if !c.raw.is_empty() {
            let raw_path = paths::data_dir().join("calibrations").join(format!(
                "{}{}.json",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
                if abort { "-cancelled" } else { "" }
            ));
            let _ = std::fs::create_dir_all(raw_path.parent().unwrap());
            let _ = std::fs::write(&raw_path, serde_json::to_vec(&serde_json::json!({ "layout": c.model.layout, "points": c.raw })).unwrap_or_default());
            println!("calibration recording saved to {}", raw_path.display());
        }
        if abort || c.model.len() < 30 {
            // the calibration may have changed the camera focal length: back to the one of the model in use
            crate::unigaze::set_focal(self.model.as_ref().and_then(|m| m.focal).unwrap_or(1000.0));
            println!("calibration cancelled/failed (is your face visible?)");
            return;
        }
        if c.model.geo.is_none() {
            let _ = c.model.fit_geometry(&self.monitors); // normally done by the calibration itself
        }
        match c.model.fit() {
            Ok(err) => {
                let _ = c.model.save(&paths::gaze_model());
                self.filter = GazeKalman::new(err * 0.5);
                println!(
                    "calibrated on {} screen(s): {} samples, cross-validated median error {err:.0}px",
                    self.monitors.len(),
                    c.model.len()
                );
                self.posture_ref = c.model.calibrated_head();
                self.moved_since = None;
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

    /// Quick recalibration: each dot is learned like a click the user surely looked at (higher weight), by both
    /// models (refit) and by the local click corrector, so the gaze is fixed at once for the current posture.
    fn finish_quick(&mut self, c: &Calibration) {
        let mons = self.monitors.clone();
        let Some(m) = &mut self.model else { return };
        const WEIGHT: f64 = 3.0;
        let t = now();
        let mut groups: Vec<i64> = c.model.g.clone();
        groups.dedup();
        let mut dots = Vec::new();
        for &g in &groups {
            let idx: Vec<usize> = (0..c.model.len()).filter(|&i| c.model.g[i] == g).collect();
            let Some(&first) = idx.first() else { continue };
            let target = c.model.y[first];
            let mut land = [0.0; NF];
            for &i in &idx {
                for k in 0..NF {
                    land[k] += c.model.f[i][k] / idx.len() as f64;
                }
            }
            let angles: Vec<Vec<f64>> = (0..c.umodel.len()).filter(|&i| c.umodel.g[i] == g).map(|i| c.umodel.f[i].clone()).collect();
            let ug = angles.last().map(|a| ([a[0], a[1]], land));
            dots.push((idx, target, angles, GazeInput { land, ug }));
        }
        let err = |m: &GazeModel, um: Option<&GazeModel>, corr: &ClickCorrector<GazeInput>, dots: &[(Vec<usize>, [f64; 2], Vec<Vec<f64>>, GazeInput)]| {
            let e: Vec<f64> = dots
                .iter()
                .filter_map(|(_, tg, _, inp)| {
                    let (x, y) = fused(m, um, inp)?;
                    let c = corr.correction([x, y], t);
                    Some((x + c[0] - tg[0]).hypot(y + c[1] - tg[1]))
                })
                .collect();
            e.iter().sum::<f64>() / e.len().max(1) as f64
        };
        let before = err(m, self.umodel.as_ref(), &self.corrector, &dots);
        for (idx, target, angles, input) in &dots {
            let feats: Vec<&Vec<f64>> = idx.iter().map(|&i| &c.model.f[i]).collect();
            m.add(&feats, *target, CLICK_GROUP + self.n_clicks as i64, WEIGHT);
            if let Some(um) = &mut self.umodel {
                um.add(angles, *target, CLICK_GROUP + self.n_clicks as i64, WEIGHT);
            }
            self.corrector.add(t, *target, input.clone());
            self.n_clicks += 1;
        }
        let _ = m.refit();
        if c.new_screens {
            // a new screen: fit its 3D pose (and refine the others) with these dots and every earlier sample
            match m.fit_geometry(&mons).and_then(|(_, what)| m.fit().map(|e| (e, what))) {
                Ok((err, what)) => println!("new screen calibrated: {what} · cross-validated error {err:.0} px"),
                Err(e) => println!("new screen calibration failed: {e:#}"),
            }
        }
        if let Some(um) = &mut self.umodel {
            let _ = um.refit();
        }
        let um = self.umodel.as_ref();
        self.corrector.update_residuals(|inp| fused(m, um, inp));
        let after = err(m, um, &self.corrector, &dots);
        let _ = m.save(&paths::gaze_model());
        if let Some(um) = um {
            let _ = um.save(&paths::gaze_model_unigaze());
        }
        self.posture_ref = Some(gaze::head_of(&c.model.f));
        self.moved_since = None;
        println!("quick recalibration on {} dots: mean error {before:.0}px -> {after:.0}px", dots.len());
        self.say(&format!("Recalibrated: error {before:.0}px -> {after:.0}px"), 4.0);
    }

    /// Notice when the user sits away from the calibrated posture (or the laptop lid moved the camera).
    fn check_posture(&mut self, t: f64, f: Option<&TrackerFrame>) {
        let (Some(fr), Some(reference)) = (f, self.posture_ref) else { return };
        let Some(feats) = fr.features.filter(|_| self.face && self.calib.is_none()) else { return };
        let shift = gaze::head_shift(reference, &feats);
        if shift > 1.0 {
            self.moved_since.get_or_insert(t);
        } else if shift < 0.7 {
            self.moved_since = None;
            self.nudges_since_move = 0;
        }
    }

    fn posture_hint(&self) -> bool {
        self.moved_since.is_some_and(|s| now() - s > 3.0)
    }

    fn say(&mut self, msg: &str, secs: f64) {
        self.notice = Some((msg.to_string(), now() + secs));
    }

    /// Copy what's under the frozen halo (selection, OCR or image) to the clipboard, in the background.
    fn capture(&mut self) {
        let Some((x, y)) = self.halo.center else { return };
        if self.capturing.swap(true, Ordering::Relaxed) {
            return;
        }
        let (r, mon) = (self.halo.radius, monitors::at(&self.monitors, x, y));
        let (tx, busy) = (self.captures.0.clone(), self.capturing.clone());
        self.say("Copying...", 10.0);
        std::thread::spawn(move || {
            // after a shortcut, wait for Ctrl / Alt to be released: held, they would change the selection drag
            {
                use device_query::{DeviceQuery, Keycode};
                let ds = device_query::DeviceState::new();
                let t0 = now();
                while now() - t0 < 3.0
                    && ds.get_keys().iter().any(|k| matches!(k, Keycode::LControl | Keycode::RControl | Keycode::LAlt | Keycode::RAlt))
                {
                    std::thread::sleep(Duration::from_millis(30));
                }
            }
            std::thread::sleep(Duration::from_millis(150)); // let the halo window disappear first
            let msg = match capture::capture(x, y, r, &mon) {
                Ok(got) => got.describe(),
                Err(e) => format!("Capture failed: {e:#}"),
            };
            println!("{msg}");
            busy.store(false, Ordering::Relaxed);
            let _ = tx.send(msg);
        });
    }

    /// PRECOG_DEBUG=1: one line per second with hands, halo and posture (for tuning gestures).
    fn debug_log(&mut self, t: f64, f: Option<&TrackerFrame>) {
        if std::env::var_os("PRECOG_DEBUG").is_none() || t - self.last_debug < 1.0 {
            return;
        }
        self.last_debug = t;
        let shift = f.and_then(|f| f.features).zip(self.posture_ref).map(|(x, r)| gaze::head_shift(r, &x));
        let (fps, hand_fps) = {
            let sh = self.tracker.shared.lock();
            (sh.fps, sh.hand_fps)
        };
        println!(
            "[debug] fps {:.0} hands {:.0} face {} gaze {:?} R {:?} L {} halo {:?} r {:.0}{} shift {}",
            fps,
            hand_fps,
            self.face,
            self.gaze.map(|g| (g.0 as i32, g.1 as i32)),
            self.pose,
            self.left.map_or("-".into(), |l| format!(
                "open {:.2} wrist ({:.2},{:.2}){}",
                l.openness,
                l.wrist[0],
                l.wrist[1],
                if l.raised { " raised" } else { "" }
            )),
            self.halo.center.map(|c| (c.0 as i32, c.1 as i32)),
            self.halo.radius,
            if self.halo.frozen { " frozen" } else { "" },
            shift.map_or("-".into(), |s| format!("{s:.2}")),
        );
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
            if self.moved_since.is_some() {
                self.nudges_since_move += 1;
                if self.nudges_since_move >= 3 {
                    // clicks have taught the gaze this posture: it becomes the reference
                    self.posture_ref = Some(gaze::head_of(&feats));
                    (self.moved_since, self.nudges_since_move) = (None, 0);
                }
            }
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
        // the open hand starts dictation only once held VOICE_HOLD (a brief open hand while gesturing triggered it)
        if raw_pose == Some(Pose::Open) {
            self.open_since.get_or_insert(t);
        } else {
            (self.open_since, self.open_used) = (None, false);
        }
        let open_held = !self.open_used && self.open_since.is_some_and(|s| t - s >= VOICE_HOLD);
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
        let dictating = self.dictation.state() != dictation::State::Idle || self.voice_starting.load(Ordering::Relaxed);
        if self.halo.frozen && !dictating {
            if left {
                self.halo.release();
                self.say("Halo released", 2.0);
                return;
            }
            if matches!(rising, Some(Pose::Fist | Pose::Pinch)) {
                self.capture();
                return;
            }
        }
        if !self.cfg.voice {
            return;
        }
        match self.dictation.state() {
            dictation::State::Idle if !self.voice_starting.load(Ordering::Relaxed) => {
                if open_held && t > self.voice_cooldown {
                    if let Some((x, y)) = self.gaze {
                        self.open_used = true;
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

    /// Fallback keyboard shortcuts (global: read from the keyboard state, whichever window has the focus), for when
    /// the gestures are hard to get: Ctrl+Alt+H freeze / release the halo, C copy under it, V voice typing (start,
    /// then send), X cancel / release, Q quit.
    fn shortcuts(&mut self, ctx: &egui::Context) {
        use device_query::{DeviceQuery, Keycode};
        let Some(ds) = &self.keys else { return };
        let keys = ds.get_keys();
        let held = |a: Keycode, b: Keycode| keys.contains(&a) || keys.contains(&b);
        let combo = (held(Keycode::LControl, Keycode::RControl) && held(Keycode::LAlt, Keycode::RAlt))
            .then(|| [Keycode::H, Keycode::C, Keycode::V, Keycode::X, Keycode::Q].into_iter().find(|k| keys.contains(k)))
            .flatten();
        if combo == self.last_combo {
            return;
        }
        self.last_combo = combo;
        if self.calib.is_some() && combo != Some(Keycode::Q) {
            return;
        }
        let t = now();
        match combo {
            Some(Keycode::H) => {
                if self.halo.frozen {
                    self.halo.release();
                    self.say("Halo released", 2.0);
                } else if self.halo.freeze() {
                    self.say("Halo frozen · Ctrl+Alt+C: copy what's under it · Ctrl+Alt+H: release", 5.0);
                }
            }
            Some(Keycode::C) => {
                if self.halo.frozen || self.halo.freeze() {
                    self.capture();
                }
            }
            Some(Keycode::V) => match self.dictation.state() {
                dictation::State::Idle if !self.voice_starting.load(Ordering::Relaxed) => {
                    if let Some((x, y)) = self.gaze {
                        self.start_voice(x, y);
                    } else {
                        self.say("Voice typing: no gaze (face not detected)", 3.0);
                    }
                }
                dictation::State::Listening => {
                    self.dictation.send();
                    self.voice_cooldown = t + 1.5;
                }
                _ => {}
            },
            Some(Keycode::X) => {
                if self.dictation.state() == dictation::State::Listening {
                    self.dictation.cancel();
                    self.voice_cooldown = t + 1.5;
                } else if self.halo.frozen {
                    self.halo.release();
                    self.say("Halo released", 2.0);
                }
            }
            Some(Keycode::Q) => ctx.send_viewport_cmd(ViewportCommand::Close),
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
        let mut parts = vec![format!("{:4.1}fps", self.tracker.shared.lock().fps), if self.face { "face ok" } else { "no face" }.into()];
        if self.cfg.gesture {
            let left = self.left.map_or("·".to_string(), |l| format!("{:.0}%", l.openness * 100.0));
            parts.push(format!("R {} L {left}", Pose::icon(self.pose)));
        }
        if self.halo.frozen {
            parts.push("[halo frozen]".into());
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
            parts.push(format!("voice {:?}", self.dictation.state()).to_lowercase());
        }
        parts.push(format!("agent: {}", if self.agent.paused { "paused".into() } else { self.agent.status.lock().clone() }));
        let hint = self.posture_hint();
        let r = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(17, 17, 17, 235))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.label(egui::RichText::new(format!("precog · {}", parts.join(" · "))).monospace().color(Color32::from_gray(220)));
                let (line, color) = match (&self.notice, hint) {
                    (Some((n, _)), _) => (n.as_str(), Color32::from_gray(230)),
                    (None, true) => ("You moved: click here for a quick recalibration (~6 s)", Color32::from_rgb(255, 204, 0)),
                    (None, false) => ("click: ask · right-click: pause · Ctrl+Alt+H/C/V/X/Q: halo/copy/voice/cancel/quit", Color32::GRAY),
                };
                ui.label(egui::RichText::new(line).small().color(color));
            });
        let resp = ui.interact(r.response.rect, egui::Id::new("hud"), egui::Sense::click());
        // Quit button on top of the HUD (registered last: it gets the clicks over it, not the HUD)
        let full = ui.max_rect();
        let quit_rect = egui::Rect::from_min_size(Pos2::new(full.max.x - 54.0, full.min.y + 6.0), egui::vec2(48.0, 20.0));
        if ui.put(quit_rect, egui::Button::new(egui::RichText::new("Quit").small())).clicked() {
            ui.ctx().send_viewport_cmd(ViewportCommand::Close);
        }
        if resp.clicked() && hint {
            self.quick_calibration();
        } else if resp.clicked() {
            self.agent.trigger(self.gaze);
        }
        if resp.secondary_clicked() {
            let mut c = self.cfg.clone();
            c.agent = !c.agent;
            self.commit(c);
        }
    }

    /// Halo around the gaze. With a compositor: Lucien's translucent smudge, sized like the halo. Without (plain X11,
    /// no transparency): a red ring, the window clipped to it. Frozen: white edge. Flashes green after a capture.
    fn halo_window(&mut self, ctx: &egui::Context) {
        let shown = self.cfg.dot && self.calib.is_none() && !self.capturing.load(Ordering::Relaxed) && (self.gaze.is_some() || self.halo.frozen);
        let Some((x, y)) = self.halo.center.filter(|_| shown) else {
            self.halo_shaped = false; // the window goes away; a new one must be shaped again
            return;
        };
        let glow = self.compositing();
        let r = self.halo.radius.round() as f32;
        let (frozen, flash) = (self.halo.frozen, now() < self.flash_until);
        let edge = match (flash, frozen) {
            (true, _) => Some((GREEN, 3.0)),
            (false, true) => Some((Color32::WHITE, 3.0)),
            (false, false) => None,
        };
        if glow {
            self.smudge.set_radius(r);
            let s = cloud::SIZE.max((r * 2.8).ceil());
            // Keep the window inside the monitor (macOS pushes windows that stick out above the screen back down,
            // which made the smudge stop short of the top) and draw the smudge offset inside it instead.
            let m = monitors::at(&self.monitors, x, y);
            let wx = (x as f32 - s / 2.0).clamp(m.left as f32, (m.right() as f32 - s).max(m.left as f32));
            let wy = (y as f32 - s / 2.0).clamp(m.top as f32, (m.bottom() as f32 - s).max(m.top as f32));
            let local = Pos2::new(x as f32 - wx, y as f32 - wy);
            let smudge = &mut self.smudge;
            ctx.show_viewport_immediate(
                ViewportId::from_hash_of("gaze"),
                overlay("precog halo", [s, s], [wx, wy]).with_mouse_passthrough(true),
                |ui, _| {
                    let ctx = ui.ctx().clone();
                    egui::CentralPanel::no_frame().show(ui, |ui| {
                        smudge.paint(&ctx, ui.painter(), local, now());
                        if let Some((color, width)) = edge {
                            ui.painter().circle_stroke(local, r, Stroke::new(width, color.gamma_multiply(0.7)));
                        }
                    });
                },
            );
            return;
        }
        let (color, width) = edge.unwrap_or((RED, 2.5));
        let s = 2.0 * r + 8.0;
        ctx.show_viewport_immediate(
            ViewportId::from_hash_of("gaze"),
            overlay("precog halo", [s, s], [x as f32 - s / 2.0, y as f32 - s / 2.0])
                .with_mouse_passthrough(true)
                .with_visible(self.halo_shaped),
            |ui, _| {
                // the window is clipped to the ring: fill it all (no anti-aliasing against a black background)
                egui::CentralPanel::no_frame().show(ui, |ui| ui.painter().rect_filled(ui.max_rect(), 0.0, color));
            },
        );
        #[cfg(target_os = "linux")]
        {
            self.halo_shaped = crate::xshape::ring("precog halo", s as u32, r, width, 0);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = width;
    }

    fn compositing(&mut self) -> bool {
        #[cfg(target_os = "linux")]
        {
            let t = now();
            if t - self.compositing.1 > 5.0 {
                self.compositing = (crate::xshape::compositing(), t);
            }
        }
        self.compositing.0
    }

    fn calibration_windows(&mut self, ctx: &egui::Context) {
        if self.calib.is_none() {
            self.preview_tex = None;
            return;
        }
        // self view: the webcam image the tracker shares while recording, as a texture (updated when it changes)
        let preview = self.tracker.shared.lock().preview.clone();
        if let Some(pv) = &preview {
            if self.preview_tex.as_ref().is_none_or(|(_, seq)| *seq != pv.seq) {
                let img = egui::ColorImage::from_rgb([pv.w, pv.h], &pv.rgb);
                match &mut self.preview_tex {
                    Some((tex, seq)) => {
                        tex.set(img, egui::TextureOptions::LINEAR);
                        *seq = pv.seq;
                    }
                    None => self.preview_tex = Some((ctx.load_texture("self-view", img, egui::TextureOptions::LINEAR), pv.seq)),
                }
            }
        }
        let view = match (&self.preview_tex, &preview) {
            (Some((tex, _)), Some(pv)) => Some(crate::calib::SelfView { tex: tex.id(), aspect: pv.w as f32 / pv.h.max(1) as f32, face: pv.face.clone() }),
            _ => None,
        };
        let Some(c) = &self.calib else { return };
        let (face, t) = (self.face, now());
        let hide = c.hide_pointer();
        let mut pointer: Option<(f64, f64)> = None;
        let mut clicks: Vec<(f64, f64)> = Vec::new();
        let (mut abort, mut redo, mut enter, mut again) = (false, false, false, false);
        for (mi, m) in self.monitors.iter().enumerate() {
            let b = ViewportBuilder::default()
                .with_title("precog calibration")
                .with_decorations(false)
                .with_always_on_top()
                .with_position([m.left as f32, m.top as f32])
                .with_inner_size([m.width as f32, m.height as f32])
                .with_taskbar(false)
                // X11: tiling WMs (i3…) float utility windows at the requested position; still managed, so ESC works
                .with_window_type(egui::X11WindowType::Utility);
            let last = self.calib_pointer;
            ctx.show_viewport_immediate(ViewportId::from_hash_of(("calib", mi)), b, |ui, _| {
                if hide {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::None);
                }
                // the pointer of this very frame (the previous one made the laser lag behind the mouse)
                let now_ptr = ui.input(|i| i.pointer.latest_pos()).map(|p| (m.left as f64 + p.x as f64, m.top as f64 + p.y as f64));
                egui::CentralPanel::no_frame().frame(Frame::new().fill(Color32::BLACK)).show(ui, |ui| {
                    c.draw(ui.painter(), mi, m, ui.max_rect().size(), face, t, now_ptr.or(last), view.as_ref());
                });
                ui.input(|i| {
                    let global = |p: Pos2| (m.left as f64 + p.x as f64, m.top as f64 + p.y as f64);
                    if let Some(p) = i.pointer.latest_pos() {
                        pointer = Some(global(p));
                    }
                    if i.pointer.primary_clicked() {
                        if let Some(p) = i.pointer.interact_pos() {
                            clicks.push(global(p));
                        }
                    }
                    abort |= i.key_pressed(egui::Key::Escape);
                    redo |= i.key_pressed(egui::Key::R);
                    enter |= i.key_pressed(egui::Key::Y) || i.key_pressed(egui::Key::Enter);
                    again |= i.key_pressed(egui::Key::A);
                });
            });
        }
        if let Some(p) = pointer {
            self.calib_pointer = Some(p);
        }
        if abort {
            self.finish_calibration(true);
            return;
        }
        let Some(c) = &mut self.calib else { return };
        if let Some((x, y)) = pointer {
            c.on_pointer(t, x, y);
        }
        if c.reviewing() {
            // the user judges the test screen; clicks are ignored there (no accidental finish)
            if redo {
                c.redo_pursuit();
            } else if again {
                self.calib = Some(Calibration::new(&self.monitors));
            } else if enter {
                self.finish_calibration(false);
            }
            return;
        }
        for (x, y) in clicks {
            c.on_click(t, x, y, &self.tracker);
        }
    }

    /// What each hand does right now, and what it will trigger: one column per hand, words (no emoji: missing from
    /// the fonts), progress bars for the gestures that need holding.
    fn gesture_window(&mut self, ctx: &egui::Context) {
        let t = now();
        if !self.cfg.gesture || self.calib.is_some() || t - self.hand_seen > 2.0 {
            return;
        }
        let (w, h) = (640.0f32, 92.0f32);
        let m = &self.monitors[0];
        let pos = [m.left as f32 + (m.width as f32 - w) / 2.0, m.bottom() as f32 - h - 24.0];
        let left = match (self.left, self.halo.frozen) {
            (_, true) => ("halo FROZEN".to_string(), "right swipe left: release · shake head: reset".to_string(), None),
            (None, false) => ("not seen".into(), "raise it (fingers up) to size and freeze the halo".into(), None),
            (Some(l), false) if !l.raised => ("lowered".into(), format!("raise it to control the halo (open {:.0}%)", l.openness * 100.0), None),
            (Some(l), false) => {
                let p = self.halo.freeze_progress(t);
                (format!("raised · open {:.0}%", l.openness * 100.0), "open/close: halo size · hold still 1 s: freeze".into(), p)
            }
        };
        let pose = match self.pose {
            Some(Pose::Open) => "OPEN",
            Some(Pose::Fist) => "FIST",
            Some(Pose::Pinch) => "PINCH",
            Some(Pose::Other) => "hand (no gesture)",
            None => "not seen",
        };
        let listening = self.dictation.state() == dictation::State::Listening;
        let right = if self.popup.is_some() {
            ("look at ACCEPT + fist or pinch: accept · swipe left: dismiss".to_string(), None)
        } else if listening {
            ("fist: send (Enter) · swipe left: cancel".into(), None)
        } else if self.halo.frozen {
            ("fist or pinch: copy what's under the halo · swipe left: release".into(), None)
        } else if self.cfg.voice {
            let p = self.open_since.filter(|_| !self.open_used).map(|s| ((t - s) / VOICE_HOLD).clamp(0.0, 1.0) as f32);
            ("hold OPEN 0.5 s: voice typing into the window you look at".into(), p)
        } else {
            ("no action (voice typing is off)".into(), None)
        };
        ctx.show_viewport_immediate(ViewportId::from_hash_of("gestures"), overlay("precog gestures", [w, h], pos), |ui, _| {
            egui::CentralPanel::no_frame().frame(Frame::new().fill(BG).inner_margin(10)).show(ui, |ui| {
                ui.columns(2, |cols| {
                    let title = |ui: &mut egui::Ui, s: &str| {
                        ui.label(egui::RichText::new(s).small().color(Color32::GRAY));
                    };
                    title(&mut cols[0], "LEFT HAND · halo");
                    cols[0].label(egui::RichText::new(&left.0).strong().color(Color32::WHITE));
                    cols[0].label(egui::RichText::new(&left.1).small().color(Color32::LIGHT_GRAY));
                    if let Some(p) = left.2 {
                        cols[0].add(egui::ProgressBar::new(p).desired_height(6.0).fill(GREEN));
                    }
                    title(&mut cols[1], "RIGHT HAND · actions");
                    cols[1].label(egui::RichText::new(pose).strong().color(Color32::WHITE));
                    cols[1].label(egui::RichText::new(&right.0).small().color(Color32::LIGHT_GRAY));
                    if let Some(p) = right.1 {
                        cols[1].add(egui::ProgressBar::new(p).desired_height(6.0).fill(GREEN));
                    }
                });
            });
        });
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
                    ui.label(egui::RichText::new(format!("{}", p.action.title)).size(18.0).strong().color(Color32::WHITE));
                    ui.label(egui::RichText::new(&p.action.reason).color(Color32::LIGHT_GRAY));
                    Frame::new().fill(CODE_BG).inner_margin(8).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(p.action.describe()).monospace().color(GREEN)).wrap());
                    });
                    ui.add_space(8.0);
                    ui.columns(2, |cols| {
                        let zones = [
                            (true, Color32::from_rgb(31, 111, 63), p.dwell_accept / need, if gestures { "ACCEPT\nlook + pinch or fist" } else { "ACCEPT\nlook here" }),
                            (false, Color32::from_rgb(90, 31, 31), p.dwell_dismiss / Popup::DISMISS_DWELL, if gestures { "DISMISS\nlook here · or swipe left" } else { "DISMISS\nlook here" }),
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
                    ui.label(egui::RichText::new("fist: send (Enter)  ·  swipe left: cancel  ·  lower hand + pause: stop").small().color(Color32::GRAY));
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
            // The landmark model is unreliable while the head moves (the eyes counter-rotate to keep the gaze
            // fixed, which it reads as gaze motion). Use its motion only while the head is still; otherwise trust
            // UniGaze alone, which models head pose.
            let w = head_still(&inp.land, land_then);
            return Some((ux + w * (lx - tx), uy + w * (ly - ty)));
        }
    }
    Some((lx, ly))
}

/// 1 when the head pose is the same in both frames, falling to 0 as it rotates (~2 deg) or moves (~1.5% of the
/// image). Feature layout: see `tracker::features` (6 pitch, 7 yaw, 8 roll, 9-10 face centre, 11 face scale).
fn head_still(a: &[f64; NF], b: &[f64; NF]) -> f64 {
    let rot = ((a[6] - b[6]).powi(2) + (a[7] - b[7]).powi(2) + (a[8] - b[8]).powi(2)).sqrt() / 0.035;
    let pos = ((a[9] - b[9]).powi(2) + (a[10] - b[10]).powi(2)).sqrt() / 0.015 + (a[11] - b[11]).abs() / 0.01;
    (-(rot * rot + pos * pos)).exp()
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
        .with_override_redirect(true) // X11: bypass the WM, tiling WMs would otherwise tile overlays
}

fn settings_form(ui: &mut egui::Ui, c: &mut Settings, custom_model: &mut String, recalibrate: &mut bool) {
    ui.heading("General");
    ui.checkbox(&mut c.agent, "Suggestions (LLM looks at the screen when your gaze settles)");
    ui.checkbox(&mut c.dot, "Show gaze halo (left hand: open/close to resize, raise and hold still to freeze)");
    ui.checkbox(&mut c.learn, "Learn from my clicks (improves gaze accuracy over time)");
    ui.add(egui::Slider::new(&mut c.dwell, 1.0..=4.0).text("fixation before asking (s)"));
    ui.add(egui::Slider::new(&mut c.interval, 5.0..=120.0).logarithmic(true).text("min time between asks (s)"));
    if ui.button("Recalibrate gaze…").clicked() {
        *recalibrate = true;
    }

    ui.separator();
    ui.heading("Gestures & voice");
    ui.checkbox(&mut c.gesture, "Hand gestures (pinch/fist accept · swipe left cancel)");
    ui.add_enabled(c.gesture, egui::Checkbox::new(&mut c.voice, "Open hand (hold 0.5 s) -> voice typing into the window I look at · fist: send"));
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
        let t0 = now();
        self.step(ctx);
        self.timing.0 = self.timing.0.max(now() - t0);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let t_ui = now();
        let ctx = ui.ctx().clone();
        // The HUD stays on top of everything: during a calibration it would hide dots, so it shrinks to 1x1 px in the
        // bottom-right corner (the dots are ~50 px from the edges). It must stay on a screen: it is the root window that
        // paces every frame, and X11 presents a window that is on no screen at a fake, ~1 Hz rate (everything lags).
        let calibrating = self.calib.is_some();
        if calibrating != self.hud_away {
            self.hud_away = calibrating;
            let p = &self.monitors[0];
            let (pos, size) = if calibrating {
                ([p.right() as f32 - 1.0, p.bottom() as f32 - 1.0], [1.0, 1.0])
            } else {
                ([p.left as f32 + 12.0, p.bottom() as f32 - 70.0], [720.0, 52.0])
            };
            ctx.send_viewport_cmd(ViewportCommand::InnerSize(egui::vec2(size[0], size[1])));
            ctx.send_viewport_cmd(ViewportCommand::OuterPosition(Pos2::new(pos[0], pos[1])));
        }
        self.hud(ui);
        self.halo_window(&ctx);
        self.calibration_windows(&ctx);
        self.popup_window(&ctx);
        self.gesture_window(&ctx);
        self.voice_window(&ctx);
        self.settings_window(&ctx);
        self.timing.1 = self.timing.1.max(now() - t_ui);
        self.timing.2 += 1;
        if std::env::var_os("PRECOG_DEBUG").is_some() && now() - self.timing.3 > 1.0 {
            println!("[timing] {} frames/s · worst logic {:.1} ms · worst ui {:.1} ms", self.timing.2, self.timing.0 * 1000.0, self.timing.1 * 1000.0);
            self.timing = (0.0, 0.0, 0, now());
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0; 4] // transparent: overlays paint their own backgrounds
    }
}
