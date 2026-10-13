//! Intent agent: when the gaze settles, screenshot (of the monitor being looked at) + gaze trail -> LLM -> one action.

use crate::actions::{self, Action, ACTION_TYPES};
use crate::monitors::{self, Monitor};
use crate::settings::Settings;
use crate::{llm, now, paths};
use anyhow::{Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use image::{imageops, Rgb, RgbImage};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const SYSTEM: &str = r#"You are Precog, a proactive desktop assistant. You see a screenshot of the user's screen.
The RED circle marks where the user is looking right now; YELLOW dots are where they looked during the last seconds
(older = smaller). Your job: anticipate the single next thing the user is about to do and offer to do it for them.

Only suggest something when you are fairly confident it is useful and specific (e.g. the user stares at an error
in a terminal -> suggest the fixing command; reads a URL / doc reference -> open it; looks at an empty Claude prompt
next to an error -> type a prompt asking Claude to fix it). Otherwise answer type "none". Being quiet is better than
being wrong. Never suggest destructive commands (rm -rf, sudo, force push, ...).

Action types:
- type_text: type `text` into the currently focused app ({app}); set `submit` true only if pressing Enter is clearly intended.
- run_command: open a new terminal and run shell `command`.
- open_url: open `url` in the browser.
- open_app: open application `app` by name.
- hotkey: press `keys`, e.g. ["cmd","t"].
- none: nothing worth suggesting.

Platform: {platform}. Focused app: {app}.
Recently suggested (avoid repeating unless the context clearly changed): {recent}

Reply with ONLY a JSON object:
{"type": "...", "title": "<=8 words, what you will do", "reason": "<=20 words, why", "text": "", "submit": false,
  "command": "", "url": "", "app": "", "keys": []}"#;

fn disc(img: &mut RgbImage, cx: f32, cy: f32, r: f32, color: Rgb<u8>, ring: Option<f32>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let r_out = r;
    let r_in = ring.map_or(0.0, |t| (r - t).max(0.0));
    for y in (cy - r_out) as i32..=(cy + r_out) as i32 {
        for x in (cx - r_out) as i32..=(cx + r_out) as i32 {
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            if d <= r_out && d >= r_in {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

pub fn screenshot_with_gaze(trail: &[(f64, f64, f64)], mon: &Monitor, t: f64) -> Result<Vec<u8>> {
    let xm = xcap::Monitor::all()?
        .into_iter()
        .find(|m| m.x().ok() == Some(mon.left) && m.y().ok() == Some(mon.top))
        .context("cannot capture the screen: grant Screen Recording to your terminal app (macOS: System Settings > Privacy & Security > Screen Recording), then restart it")?;
    let cap = xm.capture_image()?;
    let mut img: RgbImage = image::DynamicImage::ImageRgba8(cap).to_rgb8();
    let scale = img.width() as f32 / mon.width as f32; // retina: physical px per logical point
    let px = |x: f64, y: f64| (((x - mon.left as f64) as f32) * scale, ((y - mon.top as f64) as f32) * scale);
    if let Some((last, rest)) = trail.split_last() {
        for &(pt, x, y) in rest {
            let (cx, cy) = px(x, y);
            disc(&mut img, cx, cy, (10.0 - (t - pt) as f32 * 1.5).max(3.0) * scale, Rgb([255, 210, 0]), None);
        }
        let (cx, cy) = px(last.1, last.2);
        disc(&mut img, cx, cy, 45.0 * scale, Rgb([255, 0, 0]), Some(5.0 * scale));
    }
    let (w, h) = (img.width(), img.height());
    let k = (1568.0 / w.max(h) as f32).min(1.0);
    let img = imageops::resize(&img, (w as f32 * k) as u32, (h as f32 * k) as u32, imageops::FilterType::Triangle);
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80).encode_image(&img)?;
    Ok(out)
}

fn parse(text: &str) -> Result<Action> {
    let (a, b) = (text.find('{').context("no JSON in reply")?, text.rfind('}').context("no JSON in reply")?);
    Ok(serde_json::from_str(&text[a..=b])?)
}

pub fn ask(cfg: &Settings, jpeg: &[u8], app: &str, recent: &[String]) -> Result<Action> {
    let recent = if recent.is_empty() { "nothing".to_string() } else { recent.join("; ") };
    let prompt = SYSTEM
        .replace("{app}", if app.is_empty() { "unknown" } else { app })
        .replace("{platform}", std::env::consts::OS)
        .replace("{recent}", &recent);
    parse(&llm::complete(cfg, jpeg, &prompt)?)
}

/// Watches the gaze stream; on a fixation, asks the LLM in a background thread.
pub struct Agent {
    pub monitors: Vec<Monitor>,
    pub dwell_s: f64,
    pub radius: f64,
    pub min_interval: f64,
    pub paused: bool,
    pub last_call: f64,
    pub cfg: Settings,
    trail: VecDeque<(f64, f64, f64)>,
    last_fix: Option<(f64, f64)>,
    busy: Arc<AtomicBool>,
    pub status: Arc<Mutex<String>>,
    recent: Arc<Mutex<Vec<String>>>,
    tx: Sender<Action>,
    pub suggestions: Receiver<Action>,
}

impl Agent {
    pub fn new(monitors: Vec<Monitor>) -> Self {
        let (tx, suggestions) = unbounded();
        Self {
            monitors,
            dwell_s: 1.8,
            radius: 160.0,
            min_interval: 12.0,
            paused: false,
            last_call: -1e9,
            cfg: Settings::default(),
            trail: VecDeque::new(),
            last_fix: None,
            busy: Arc::new(AtomicBool::new(false)),
            status: Arc::new(Mutex::new("idle".into())),
            recent: Arc::new(Mutex::new(Vec::new())),
            tx,
            suggestions,
        }
    }

    pub fn busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }

    pub fn update(&mut self, t: f64, x: f64, y: f64) {
        self.trail.push_back((t, x, y));
        while self.trail.front().is_some_and(|p| t - p.0 > 10.0) {
            self.trail.pop_front();
        }
        if self.busy() || self.paused || t - self.last_call < self.min_interval {
            return;
        }
        let pts: Vec<(f64, f64)> = self.trail.iter().filter(|p| t - p.0 <= self.dwell_s).map(|p| (p.1, p.2)).collect();
        if pts.len() < 10 || t - self.trail[0].0 < self.dwell_s {
            return;
        }
        let n = pts.len() as f64;
        let (cx, cy) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
        if pts.iter().any(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() > self.radius) {
            return; // still scanning, not fixating
        }
        if let Some((lx, ly)) = self.last_fix {
            if ((cx - lx).powi(2) + (cy - ly).powi(2)).sqrt() < self.radius && t - self.last_call < 4.0 * self.min_interval {
                return; // same spot as last time
            }
        }
        self.trigger(Some((cx, cy)));
    }

    pub fn trigger(&mut self, fix: Option<(f64, f64)>) {
        if self.busy() {
            return;
        }
        let t = now();
        self.busy.store(true, Ordering::Relaxed);
        self.last_call = t;
        self.last_fix = fix;
        let trail: Vec<_> = self.trail.iter().filter(|p| t - p.0 <= 6.0).copied().collect();
        let (x, y) = trail.last().map(|p| (p.1, p.2)).or(fix).unwrap_or((self.monitors[0].left as f64, self.monitors[0].top as f64));
        let mon = monitors::at(&self.monitors, x, y);
        let (cfg, busy, status, recent, tx) =
            (self.cfg.clone(), self.busy.clone(), self.status.clone(), self.recent.clone(), self.tx.clone());
        std::thread::spawn(move || {
            *status.lock() = "thinking…".into();
            let res = (|| -> Result<Option<(Action, f64)>> {
                let app = actions::frontmost_app();
                let jpeg = screenshot_with_gaze(&trail, &mon, t)?;
                if std::env::var_os("PRECOG_DEBUG").is_some() {
                    let _ = std::fs::create_dir_all(paths::cache_dir());
                    let _ = std::fs::write(paths::cache_dir().join("last_screen.jpg"), &jpeg);
                }
                let t0 = now();
                let rec = recent.lock().clone();
                let mut a = ask(&cfg, &jpeg, &app, &rec)?;
                let dt = now() - t0;
                if !ACTION_TYPES.contains(&a.kind.as_str()) || actions::is_dangerous(&a) {
                    return Ok(Some((Action { kind: "none".into(), ..Default::default() }, dt)));
                }
                a.target_app = app;
                Ok(Some((a, dt)))
            })();
            *status.lock() = match res {
                Ok(Some((a, dt))) if a.kind != "none" => {
                    let mut r = recent.lock();
                    r.push(format!("{}: {}", a.kind, a.describe()));
                    if r.len() > 6 {
                        r.remove(0);
                    }
                    let _ = tx.send(a);
                    format!("suggested ({dt:.1}s)")
                }
                Ok(Some((_, dt))) => format!("nothing to suggest ({dt:.1}s)"),
                Ok(None) => "idle".into(),
                Err(e) => format!("error: {e:#}").chars().take(120).collect(),
            };
            busy.store(false, Ordering::Relaxed);
        });
    }
}
