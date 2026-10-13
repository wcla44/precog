//! Menu-bar / system-tray icon. Quick toggles here; everything else lives in the Settings window.
//!
//! macOS: created on the main thread inside the eframe event loop.
//! Linux: tray-icon needs GTK, so the icon lives on its own thread running the GTK main loop.

use crate::settings::Settings;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    Toggle(&'static str), // settings key
    Ask,
    Calibrate,
    Settings,
    Quit,
}

#[derive(Clone)]
struct Ids {
    map: Vec<(MenuId, Cmd)>,
}

struct Items {
    status: MenuItem,
    checks: Vec<(&'static str, CheckMenuItem)>,
    icon: TrayIcon,
}

enum Update {
    Status(String),
    Settings(Settings),
}

pub struct Tray {
    ids: Ids,
    #[cfg(target_os = "macos")]
    items: Items,
    #[cfg(not(target_os = "macos"))]
    tx: std::sync::mpsc::Sender<Update>,
}

fn eye_icon(active: bool) -> Icon {
    let s = 44u32;
    let mut px = vec![0u8; (s * s * 4) as usize];
    let c = s as f32 / 2.0;
    for y in 0..s {
        for x in 0..s {
            let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
            // almond eye outline: two circle arcs; pupil in the middle
            let e = (dx / 19.0).powi(2) + (dy / 11.0).powi(2);
            let ring = (0.72..=1.0).contains(&e);
            let pupil = dx * dx + dy * dy <= 30.0;
            let slash = !active && (dx + dy).abs() < 2.5 && dx.abs() < 19.0;
            if ring || pupil || slash {
                let i = ((y * s + x) * 4) as usize;
                px[i..i + 4].copy_from_slice(&[0, 0, 0, if active { 255 } else { 150 }]);
            }
        }
    }
    Icon::from_rgba(px, s, s).expect("icon")
}

fn build(cfg: &Settings) -> anyhow::Result<(Items, Ids)> {
    let status = MenuItem::new("precog: starting…", false, None);
    let check = |label: &str, on: bool| CheckMenuItem::new(label, true, on, None);
    let checks: Vec<(&'static str, CheckMenuItem)> = vec![
        ("agent", check("Suggestions on", cfg.agent)),
        ("dot", check("Show gaze cloud", cfg.dot)),
        ("gesture", check("Hand gestures", cfg.gesture)),
        ("voice", check("✋ Open hand → voice typing", cfg.voice)),
        ("learn", check("Learn from my clicks", cfg.learn)),
    ];
    let ask = MenuItem::new("Ask now", true, None);
    let calibrate = MenuItem::new("Recalibrate (train)…", true, None);
    let settings = MenuItem::new("Settings…", true, None);
    let quit = MenuItem::new("Quit precog", true, None);
    let hint = MenuItem::new("✊ fist: send · 👈 swipe left: cancel", false, None);

    let menu = Menu::new();
    menu.append(&status)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&checks[0].1)?;
    menu.append(&ask)?;
    menu.append(&PredefinedMenuItem::separator())?;
    for (_, c) in &checks[1..] {
        menu.append(c)?;
    }
    menu.append(&hint)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&calibrate)?;
    menu.append(&settings)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&quit)?;

    let mut map: Vec<(MenuId, Cmd)> = checks.iter().map(|(k, c)| (c.id().clone(), Cmd::Toggle(k))).collect();
    map.push((ask.id().clone(), Cmd::Ask));
    map.push((calibrate.id().clone(), Cmd::Calibrate));
    map.push((settings.id().clone(), Cmd::Settings));
    map.push((quit.id().clone(), Cmd::Quit));

    let icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_icon_templated(eye_icon(cfg.agent))
        .with_tooltip("precog")
        .build()?;
    Ok((Items { status, checks, icon }, Ids { map }))
}

impl Items {
    fn apply(&self, u: Update) {
        match u {
            Update::Status(s) => self.status.set_text(s),
            Update::Settings(cfg) => {
                for (k, c) in &self.checks {
                    c.set_checked(get_bool(&cfg, k));
                }
                let _ = self.icon.set_icon_templated(Some(eye_icon(cfg.agent)));
            }
        }
    }
}

pub fn get_bool(cfg: &Settings, key: &str) -> bool {
    match key {
        "agent" => cfg.agent,
        "dot" => cfg.dot,
        "gesture" => cfg.gesture,
        "voice" => cfg.voice,
        "learn" => cfg.learn,
        _ => false,
    }
}

pub fn set_bool(cfg: &mut Settings, key: &str, v: bool) {
    match key {
        "agent" => cfg.agent = v,
        "dot" => cfg.dot = v,
        "gesture" => cfg.gesture = v,
        "voice" => cfg.voice = v,
        "learn" => cfg.learn = v,
        _ => {}
    }
}

impl Tray {
    #[cfg(target_os = "macos")]
    pub fn new(cfg: &Settings) -> anyhow::Result<Self> {
        let (items, ids) = build(cfg)?;
        Ok(Self { ids, items })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new(cfg: &Settings) -> anyhow::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<Update>();
        let (id_tx, id_rx) = std::sync::mpsc::channel();
        let cfg = cfg.clone();
        std::thread::spawn(move || {
            if gtk::init().is_err() {
                let _ = id_tx.send(Err(anyhow::anyhow!("GTK init failed")));
                return;
            }
            let (items, ids) = match build(&cfg) {
                Ok(v) => v,
                Err(e) => {
                    let _ = id_tx.send(Err(e));
                    return;
                }
            };
            let _ = id_tx.send(Ok(ids));
            gtk::glib::timeout_add_local(std::time::Duration::from_millis(200), move || {
                while let Ok(u) = rx.try_recv() {
                    items.apply(u);
                }
                gtk::glib::ControlFlow::Continue
            });
            gtk::main();
        });
        let ids = id_rx.recv()??;
        Ok(Self { ids, tx })
    }

    fn send(&self, u: Update) {
        #[cfg(target_os = "macos")]
        self.items.apply(u);
        #[cfg(not(target_os = "macos"))]
        let _ = self.tx.send(u);
    }

    pub fn set_status(&self, s: &str) {
        self.send(Update::Status(s.to_string()));
    }

    pub fn sync(&self, cfg: &Settings) {
        self.send(Update::Settings(cfg.clone()));
    }

    /// Menu clicks since last call.
    pub fn poll(&self) -> Vec<Cmd> {
        let mut out = Vec::new();
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if let Some((_, c)) = self.ids.map.iter().find(|(id, _)| *id == ev.id) {
                out.push(*c);
            }
        }
        out
    }
}
