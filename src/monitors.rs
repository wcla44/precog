//! Monitor layout in global logical coordinates (points on macOS, pixels on X11), top-left origin.
//! Same space as window positions, mouse coordinates and Quartz window bounds.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Monitor {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub primary: bool,
    /// Physical size (mm) as reported by the screen (EDID), when it does: context for the gaze layout.
    pub mm: Option<(f64, f64)>,
    /// The laptop's own panel (eDP, LVDS, DSI; macOS: built-in display): the webcam sits in its bezel.
    pub builtin: bool,
}

impl Monitor {
    pub fn right(&self) -> i32 {
        self.left + self.width as i32
    }
    pub fn bottom(&self) -> i32 {
        self.top + self.height as i32
    }
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.left as f64 && x < self.right() as f64 && y >= self.top as f64 && y < self.bottom() as f64
    }
    pub fn clamp(&self, x: f64, y: f64) -> (f64, f64) {
        (x.clamp(self.left as f64, (self.right() - 1) as f64), y.clamp(self.top as f64, (self.bottom() - 1) as f64))
    }
    pub fn layout(&self) -> (i32, i32, u32, u32) {
        (self.left, self.top, self.width, self.height)
    }
}

/// The screen whose bezel holds the webcam: the built-in panel, else the primary screen. Returned first, the others
/// in their usual order (the geometric gaze model anchors the camera to the first screen).
pub fn camera_first(mons: &[Monitor]) -> Vec<Monitor> {
    let cam = mons.iter().position(|m| m.builtin).unwrap_or(0);
    std::iter::once(mons[cam]).chain(mons.iter().enumerate().filter(|(i, _)| *i != cam).map(|(_, m)| *m)).collect()
}

/// Primary monitor first.
pub fn list() -> Vec<Monitor> {
    let mut out = platform_list();
    out.sort_by_key(|m| !m.primary);
    if out.is_empty() {
        out.push(Monitor { left: 0, top: 0, width: 1440, height: 900, primary: true, mm: None, builtin: true });
    }
    out
}

/// macOS: Quartz display bounds (points, global top-left origin); needs no Screen Recording permission.
#[cfg(target_os = "macos")]
fn platform_list() -> Vec<Monitor> {
    use core_graphics::display::CGDisplay;
    CGDisplay::active_displays()
        .unwrap_or_default()
        .into_iter()
        .map(|id| {
            let d = CGDisplay::new(id);
            let b = d.bounds();
            let mm = d.screen_size();
            Monitor {
                left: b.origin.x as i32,
                top: b.origin.y as i32,
                width: b.size.width as u32,
                height: b.size.height as u32,
                primary: d.is_main(),
                mm: (mm.width > 0.0 && mm.height > 0.0).then_some((mm.width, mm.height)),
                builtin: d.is_builtin(),
            }
        })
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn platform_list() -> Vec<Monitor> {
    let mm = randr_sizes();
    xcap::Monitor::all()
        .unwrap_or_default()
        .iter()
        .filter_map(|m| {
            let (left, top, width, height) = (m.x().ok()?, m.y().ok()?, m.width().ok()?, m.height().ok()?);
            Some(Monitor {
                left,
                top,
                width,
                height,
                primary: m.is_primary().unwrap_or(false),
                mm: mm.iter().find(|(r, ..)| *r == (left, top, width, height)).map(|(_, s, _)| *s),
                builtin: mm.iter().find(|(r, ..)| *r == (left, top, width, height)).is_some_and(|(.., b)| *b),
            })
        })
        .collect()
}

/// XRandR: physical size (mm) of each active output and whether it is the built-in panel, keyed by its rect.
#[cfg(not(target_os = "macos"))]
fn randr_sizes() -> Vec<((i32, i32, u32, u32), (f64, f64), bool)> {
    use x11rb::connection::Connection;
    use x11rb::protocol::randr::ConnectionExt;
    let Ok((conn, screen)) = x11rb::connect(None) else { return Vec::new() };
    let root = conn.setup().roots[screen].root;
    let Some(res) = conn.randr_get_screen_resources_current(root).ok().and_then(|c| c.reply().ok()) else { return Vec::new() };
    res.outputs
        .iter()
        .filter_map(|&o| {
            let out = conn.randr_get_output_info(o, res.config_timestamp).ok()?.reply().ok()?;
            if out.crtc == 0 || out.mm_width == 0 || out.mm_height == 0 {
                return None;
            }
            let crtc = conn.randr_get_crtc_info(out.crtc, res.config_timestamp).ok()?.reply().ok()?;
            let name = String::from_utf8_lossy(&out.name).to_ascii_lowercase();
            let builtin = ["edp", "lvds", "dsi"].iter().any(|p| name.starts_with(p));
            Some(((crtc.x as i32, crtc.y as i32, crtc.width as u32, crtc.height as u32), (out.mm_width as f64, out.mm_height as f64), builtin))
        })
        .collect()
}

pub fn at(monitors: &[Monitor], x: f64, y: f64) -> Monitor {
    if let Some(m) = monitors.iter().find(|m| m.contains(x, y)) {
        return *m;
    }
    *monitors
        .iter()
        .min_by(|a, b| {
            let da = dist2(a, x, y);
            let db = dist2(b, x, y);
            da.total_cmp(&db)
        })
        .unwrap()
}

/// Snap a point that fell between/outside screens onto the nearest one.
pub fn clamp(monitors: &[Monitor], x: f64, y: f64) -> (f64, f64) {
    at(monitors, x, y).clamp(x, y)
}

fn dist2(m: &Monitor, x: f64, y: f64) -> f64 {
    let (cx, cy) = m.clamp(x, y);
    (cx - x).powi(2) + (cy - y).powi(2)
}
