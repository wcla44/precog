//! Capture what is under the frozen halo into the clipboard:
//! - Linux, text apps (terminals, browsers, editors): a real selection dragged across the halo, read back from the
//!   X11 PRIMARY selection, so the text is exact;
//! - otherwise OCR (the `tesseract` CLI, if installed) of a screenshot of the zone, if it holds text;
//! - otherwise the screenshot itself, as an image.

use crate::monitors::Monitor;
use crate::paths;
use anyhow::{Context, Result};
use image::RgbaImage;
use parking_lot::{Mutex, MutexGuard};
use std::process::Command;
use std::sync::OnceLock;

pub enum Captured {
    Text { text: String, how: &'static str },
    Image { w: u32, h: u32 },
}

impl Captured {
    pub fn describe(&self) -> String {
        match self {
            Captured::Text { text, how } => format!("Copied {} chars ({how})", text.chars().count()),
            Captured::Image { w, h } => format!("Copied a {w}x{h} image"),
        }
    }
}

/// Window classes where dragging selects text (lowercase substrings of WM_CLASS).
#[cfg(target_os = "linux")]
const TEXT_APPS: &[&str] = &[
    "term", "ghostty", "kitty", "alacritty", "wezterm", "foot", "konsole", "tilix", "terminator", "urxvt", "firefox",
    "chrom", "brave", "vivaldi", "opera", "code", "codium", "cursor", "zed", "emacs", "vim", "sublime", "jetbrains",
    "gedit", "kate", "mousepad", "obsidian", "libreoffice",
];

pub fn capture(x: f64, y: f64, r: f64, mon: &Monitor) -> Result<Captured> {
    #[cfg(target_os = "linux")]
    {
        use enigo::{Coordinate, Mouse};
        let mut mouse = crate::actions::enigo()?;
        let back = mouse.location().unwrap_or((x as i32, y as i32));
        let selected = text_app_at(x, y).map(|app| (select_text(&mut mouse, x, y, r), app));
        let _ = mouse.move_mouse(back.0, back.1, Coordinate::Abs);
        match selected {
            Some((Ok(Some(text)), _)) => {
                clipboard()?.set_text(text.clone())?;
                return Ok(Captured::Text { text, how: "selection" });
            }
            Some((Ok(None), app)) => println!("nothing selected in {app}: falling back to OCR"),
            Some((Err(e), app)) => println!("selection in {app} failed ({e:#}): falling back to OCR"),
            None => {}
        }
    }
    let img = grab(x, y, r, mon)?;
    if let Some(text) = ocr(&img) {
        clipboard()?.set_text(text.clone())?;
        return Ok(Captured::Text { text, how: "OCR" });
    }
    let (w, h) = img.dimensions();
    clipboard()?.set_image(arboard::ImageData { width: w as usize, height: h as usize, bytes: img.into_raw().into() })?;
    Ok(Captured::Image { w, h })
}

/// One clipboard for the whole run: on X11 the owning process serves the copied content to other apps.
fn clipboard() -> Result<MutexGuard<'static, arboard::Clipboard>> {
    static CLIP: OnceLock<Mutex<arboard::Clipboard>> = OnceLock::new();
    if CLIP.get().is_none() {
        let c = arboard::Clipboard::new().context("clipboard unavailable")?;
        let _ = CLIP.set(Mutex::new(c));
    }
    Ok(CLIP.get().unwrap().lock())
}

/// Square around the halo, in physical pixels of `mon`.
fn grab(x: f64, y: f64, r: f64, mon: &Monitor) -> Result<RgbaImage> {
    let xm = xcap::Monitor::all()?
        .into_iter()
        .find(|m| m.x().ok() == Some(mon.left) && m.y().ok() == Some(mon.top))
        .context("cannot capture the screen (macOS: grant Screen Recording)")?;
    let img = xm.capture_image()?;
    let scale = img.width() as f64 / mon.width as f64; // retina: physical px per logical point
    let clip = |v: f64, max: u32| (v * scale).clamp(0.0, max as f64) as u32;
    let (x0, y0) = (clip(x - r - mon.left as f64, img.width()), clip(y - r - mon.top as f64, img.height()));
    let (x1, y1) = (clip(x + r - mon.left as f64, img.width()), clip(y + r - mon.top as f64, img.height()));
    anyhow::ensure!(x1 > x0 && y1 > y0, "halo is off screen");
    let mut out = image::imageops::crop_imm(&img, x0, y0, x1 - x0, y1 - y0).to_image();
    for p in out.pixels_mut() {
        p.0[3] = 255; // X11 captures come with alpha 0: the image would paste fully transparent
    }
    Ok(out)
}

/// Text in the image, or None if tesseract is missing or finds nothing it is confident about.
fn ocr(img: &RgbaImage) -> Option<String> {
    // screen text is small: tesseract works best with glyphs ~2x larger
    let gray = image::DynamicImage::ImageLuma8(image::DynamicImage::ImageRgba8(img.clone()).to_luma8());
    let big = gray.resize(img.width() * 2, img.height() * 2, image::imageops::FilterType::Triangle);
    let dir = paths::cache_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("capture-{}.png", std::process::id()));
    big.save(&path).ok()?;
    let out = Command::new("tesseract").arg(&path).arg("stdout").args(["-l", ocr_langs(), "--psm", "6", "tsv"]).output();
    let _ = std::fs::remove_file(&path);
    let out = out.ok().filter(|o| o.status.success())?;
    parse_tsv(&String::from_utf8_lossy(&out.stdout))
}

/// English, plus French when its data is installed.
fn ocr_langs() -> &'static str {
    static LANGS: OnceLock<&'static str> = OnceLock::new();
    LANGS.get_or_init(|| {
        let out = Command::new("tesseract").arg("--list-langs").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        if out.unwrap_or_default().lines().any(|l| l.trim() == "fra") {
            "eng+fra"
        } else {
            "eng"
        }
    })
}

/// tesseract TSV -> text, line by line; None unless the words are confident enough to be real text.
fn parse_tsv(tsv: &str) -> Option<String> {
    let mut lines: Vec<((&str, &str, &str), Vec<&str>)> = Vec::new();
    let mut confs = Vec::new();
    for row in tsv.lines().skip(1) {
        let c: Vec<&str> = row.split('\t').collect();
        if c.len() < 12 || c[0] != "5" {
            continue; // level 5 = word
        }
        let (conf, word) = (c[10].parse::<f32>().unwrap_or(-1.0), c[11].trim());
        if conf < 0.0 || word.is_empty() {
            continue;
        }
        confs.push(conf);
        let key = (c[2], c[3], c[4]); // block, paragraph, line
        match lines.last_mut() {
            Some((k, words)) if *k == key => words.push(word),
            _ => lines.push((key, vec![word])),
        }
    }
    let mean = confs.iter().sum::<f32>() / confs.len().max(1) as f32;
    let text: Vec<String> = lines.into_iter().map(|(_, w)| w.join(" ")).collect();
    let text = text.join("\n");
    let alnum = text.chars().filter(|c| c.is_alphanumeric()).count();
    (mean >= 60.0 && alnum >= 3).then_some(text)
}

/// Class of the window under (x, y), if dragging selects text there. Moves the mouse there: the window that would
/// get the click is the one under the pointer (window lists also hold windows of hidden workspaces).
#[cfg(target_os = "linux")]
fn text_app_at(x: f64, y: f64) -> Option<String> {
    let sh = |args: &[&str]| Command::new("xdotool").args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    let loc = sh(&["mousemove", &(x as i64).to_string(), &(y as i64).to_string(), "getmouselocation", "--shell"])?;
    let wid = loc.lines().find_map(|l| l.strip_prefix("WINDOW="))?.trim().to_string();
    let class = sh(&["getwindowclassname", &wid])?.trim().to_lowercase();
    TEXT_APPS.iter().any(|a| class.contains(a)).then_some(class)
}

/// Drag a selection across the halo, then read the X11 PRIMARY selection. None if the selection didn't change.
#[cfg(target_os = "linux")]
fn select_text(e: &mut enigo::Enigo, x: f64, y: f64, r: f64) -> Result<Option<String>> {
    use arboard::{GetExtLinux, LinuxClipboardKind};
    use enigo::{Button, Coordinate, Direction, Mouse};
    use std::time::Duration;
    let primary = || clipboard().ok().and_then(|mut c| c.get().clipboard(LinuxClipboardKind::Primary).text().ok()).unwrap_or_default();
    let before = primary();
    // from the top-left of the halo's inner rectangle to its bottom-right: whole lines in between
    let (x0, y0, x1, y1) = (x - 0.9 * r, y - 0.6 * r, x + 0.9 * r, y + 0.6 * r);
    e.move_mouse(x0 as i32, y0 as i32, Coordinate::Abs)?;
    std::thread::sleep(Duration::from_millis(40));
    e.button(Button::Left, Direction::Press)?;
    for i in 1..=12 {
        let f = i as f64 / 12.0;
        e.move_mouse((x0 + f * (x1 - x0)) as i32, (y0 + f * (y1 - y0)) as i32, Coordinate::Abs)?;
        std::thread::sleep(Duration::from_millis(12));
    }
    e.button(Button::Left, Direction::Release)?;
    std::thread::sleep(Duration::from_millis(150));
    let after = primary();
    Ok((!after.trim().is_empty() && after != before).then_some(after))
}

#[cfg(test)]
mod tests {
    #[test]
    fn tsv() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t91\tHello\n5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t88\tworld\n\
                   5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t90\tagain\n";
        assert_eq!(super::parse_tsv(tsv).as_deref(), Some("Hello world\nagain"));
        assert_eq!(super::parse_tsv("header\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t20\t~;\n"), None);
    }
}
