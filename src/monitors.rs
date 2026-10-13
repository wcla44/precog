//! Monitor layout in global logical coordinates (points on macOS, pixels on X11), top-left origin.
//! Same space as window positions, mouse coordinates and Quartz window bounds.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Monitor {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub primary: bool,
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

/// Primary monitor first.
pub fn list() -> Vec<Monitor> {
    let mut out = platform_list();
    out.sort_by_key(|m| !m.primary);
    if out.is_empty() {
        out.push(Monitor { left: 0, top: 0, width: 1440, height: 900, primary: true });
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
            Monitor {
                left: b.origin.x as i32,
                top: b.origin.y as i32,
                width: b.size.width as u32,
                height: b.size.height as u32,
                primary: d.is_main(),
            }
        })
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn platform_list() -> Vec<Monitor> {
    xcap::Monitor::all()
        .unwrap_or_default()
        .iter()
        .filter_map(|m| {
            Some(Monitor {
                left: m.x().ok()?,
                top: m.y().ok()?,
                width: m.width().ok()?,
                height: m.height().ok()?,
                primary: m.is_primary().unwrap_or(false),
            })
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
