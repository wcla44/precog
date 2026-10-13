//! X11 without a compositor has no window transparency: an overlay's transparent background shows as a black
//! square. The SHAPE extension clips the window itself, so the halo becomes just its ring.

use parking_lot::Mutex;
use std::sync::OnceLock;
use x11rb::connection::Connection;
use x11rb::protocol::shape::{self, SK, SO};
use x11rb::protocol::xproto::{AtomEnum, ClipOrdering, ConnectionExt, Rectangle, Window};
use x11rb::rust_connection::RustConnection;

struct X {
    conn: RustConnection,
    root: Window,
    screen: usize,
    /// (title, window, shape key) of the last shaped window.
    shaped: Option<(String, Window, u64)>,
}

fn x() -> Option<&'static Mutex<X>> {
    static X: OnceLock<Option<Mutex<X>>> = OnceLock::new();
    X.get_or_init(|| {
        let (conn, screen) = x11rb::connect(None).ok()?;
        let root = conn.setup().roots[screen].root;
        Some(Mutex::new(X { conn, root, screen, shaped: None }))
    })
    .as_ref()
}

/// A compositing manager owns _NET_WM_CM_Sn: then windows can really be translucent.
pub fn compositing() -> bool {
    let Some(x) = x() else { return false };
    let x = x.lock();
    let name = format!("_NET_WM_CM_S{}", x.screen);
    let Some(atom) = x.conn.intern_atom(false, name.as_bytes()).ok().and_then(|c| c.reply().ok()) else { return false };
    x.conn.get_selection_owner(atom.atom).ok().and_then(|c| c.reply().ok()).is_some_and(|r| r.owner != x11rb::NONE)
}

/// Clip our top-level window titled `title` (a `size`x`size` square) to a ring of radius `r` and width `width`,
/// drawn with `dash` (dash count around the circle, 0 = solid). Cheap when nothing changed. Returns true once the
/// window has that shape (it may not exist yet right after the viewport is created).
pub fn ring(title: &str, size: u32, r: f32, width: f32, dashes: u32) -> bool {
    let Some(x) = x() else { return false };
    let mut x = x.lock();
    let key = ((size as u64) << 40) | ((r * 4.0) as u64) << 20 | ((width * 4.0) as u64) << 8 | dashes as u64;
    // our override-redirect overlays are direct children of the root window
    let window = match &x.shaped {
        Some((t, w, _)) if t == title && x.conn.get_geometry(*w).ok().and_then(|c| c.reply().ok()).is_some() => *w,
        _ => match find(&x, title) {
            Some(w) => w,
            None => return false,
        },
    };
    if x.shaped.as_ref().is_some_and(|(t, w, k)| t == title && *w == window && *k == key) {
        return true;
    }
    let rects = ring_spans(size, r, width, dashes);
    let ok = shape::rectangles(&x.conn, SO::SET, SK::BOUNDING, ClipOrdering::UNSORTED, window, 0, 0, &rects)
        .is_ok_and(|c| c.check().is_ok());
    if ok {
        x.shaped = Some((title.to_string(), window, key));
    }
    ok
}

fn find(x: &X, title: &str) -> Option<Window> {
    let tree = x.conn.query_tree(x.root).ok()?.reply().ok()?;
    tree.children.into_iter().rev().find(|&w| {
        x.conn
            .get_property(false, w, AtomEnum::WM_NAME, AtomEnum::ANY, 0, 256)
            .ok()
            .and_then(|c| c.reply().ok())
            .is_some_and(|p| p.value == title.as_bytes())
    })
}

/// The ring as horizontal 1-pixel spans: pixels whose distance to the centre is within width/2 of r, and that fall
/// on a dash.
fn ring_spans(size: u32, r: f32, width: f32, dashes: u32) -> Vec<Rectangle> {
    let c = size as f32 / 2.0;
    let (r_in, r_out) = ((r - width / 2.0).max(0.0), r + width / 2.0);
    let on = |px: f32, py: f32| {
        let d = (px - c).hypot(py - c);
        if d < r_in || d > r_out {
            return false;
        }
        if dashes == 0 {
            return true;
        }
        let a = (py - c).atan2(px - c) / std::f32::consts::TAU + 0.5; // 0..1 around the circle
        (a * dashes as f32).fract() < 0.6
    };
    let mut rects = Vec::new();
    for y in 0..size {
        let py = y as f32 + 0.5;
        if (py - c).abs() > r_out {
            continue;
        }
        let mut run: Option<u32> = None;
        for xx in 0..=size {
            let inside = xx < size && on(xx as f32 + 0.5, py);
            match (inside, run) {
                (true, None) => run = Some(xx),
                (false, Some(x0)) => {
                    rects.push(Rectangle { x: x0 as i16, y: y as i16, width: (xx - x0) as u16, height: 1 });
                    run = None;
                }
                _ => {}
            }
        }
    }
    rects
}

#[cfg(test)]
mod tests {
    #[test]
    fn ring_area() {
        let (r, w) = (100.0, 4.0);
        let area: u32 = super::ring_spans(260, r, w, 0).iter().map(|s| s.width as u32).sum();
        let expected = std::f32::consts::TAU * r * w;
        assert!((area as f32 - expected).abs() / expected < 0.05, "{area} vs {expected}");
        let dashed: u32 = super::ring_spans(260, r, w, 24).iter().map(|s| s.width as u32).sum();
        assert!((dashed as f32 / area as f32 - 0.6).abs() < 0.05);
    }
}
