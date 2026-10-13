//! Actions the LLM may suggest, and focusing the window under the user's gaze. macOS + Linux (X11).

use anyhow::{bail, Result};
use enigo::{Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings as EnigoSettings};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Duration;

const MAC: bool = cfg!(target_os = "macos");

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Action {
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    pub reason: String,
    pub text: String,
    pub submit: bool,
    pub command: String,
    pub url: String,
    pub app: String,
    pub keys: Vec<String>,
    /// App that was focused when the suggestion was made (where to type).
    #[serde(skip)]
    pub target_app: String,
}

pub const ACTION_TYPES: &[&str] = &["type_text", "run_command", "open_url", "open_app", "hotkey"];

pub fn is_dangerous(a: &Action) -> bool {
    let s = format!("{} {}", a.command, a.text).to_lowercase();
    let pats = ["rm -r", "rm -fr", "sudo ", "mkfs", "dd if=", "> /dev/sd", "shutdown", "reboot", ":(){", "push --force", "push -f"];
    pats.iter().any(|p| s.contains(p)) || (s.contains("curl") && (s.contains("| sh") || s.contains("| bash") || s.contains("|sh")))
}

impl Action {
    pub fn describe(&self) -> String {
        match self.kind.as_str() {
            "type_text" => format!("type: {}{}", self.text, if self.submit { "  ⏎" } else { "" }),
            "run_command" => format!("$ {}", self.command),
            "open_url" => self.url.clone(),
            "open_app" => format!("open {}", self.app),
            "hotkey" => self.keys.join("+"),
            k => k.to_string(),
        }
    }
}

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn osascript(script: &str) -> String {
    sh("osascript", &["-e", script])
}

pub fn frontmost_app() -> String {
    if MAC {
        osascript(r#"tell application "System Events" to get name of first application process whose frontmost is true"#)
    } else {
        sh("xdotool", &["getactivewindow", "getwindowname"])
    }
}

pub fn activate(app: &str) {
    if app.is_empty() {
        return;
    }
    if MAC {
        osascript(&format!(r#"tell application "{}" to activate"#, app.replace('"', "")));
    } else {
        sh("xdotool", &["search", "--name", app, "windowactivate"]);
    }
    std::thread::sleep(Duration::from_millis(350));
}

pub fn enigo() -> Result<Enigo> {
    Ok(Enigo::new(&EnigoSettings::default())?)
}

fn key_of(name: &str) -> Key {
    match name.to_lowercase().as_str() {
        "cmd" | "command" | "super" | "meta" => Key::Meta,
        "ctrl" | "control" => Key::Control,
        "alt" | "option" => Key::Alt,
        "shift" => Key::Shift,
        "enter" | "return" => Key::Return,
        "tab" => Key::Tab,
        "esc" | "escape" => Key::Escape,
        "space" => Key::Space,
        "up" => Key::UpArrow,
        "down" => Key::DownArrow,
        "left" => Key::LeftArrow,
        "right" => Key::RightArrow,
        "backspace" => Key::Backspace,
        s => Key::Unicode(s.chars().next().unwrap_or(' ')),
    }
}

pub fn execute(a: &Action) -> Result<()> {
    match a.kind.as_str() {
        "type_text" => {
            activate(&a.target_app);
            let mut e = enigo()?;
            e.text(&a.text)?;
            if a.submit {
                e.key(Key::Return, Direction::Click)?;
            }
        }
        "run_command" => {
            if MAC {
                let esc = a.command.replace('\\', "\\\\").replace('"', "\\\"");
                osascript(&format!("tell application \"Terminal\" to do script \"{esc}\""));
                osascript("tell application \"Terminal\" to activate");
            } else {
                let terms: [(&str, &[&str]); 4] =
                    [("gnome-terminal", &["--"]), ("konsole", &["-e"]), ("x-terminal-emulator", &["-e"]), ("xterm", &["-e"])];
                let script = format!("{}; exec bash", a.command);
                let ok = terms.iter().any(|(t, pre)| {
                    Command::new(t).args(*pre).args(["bash", "-c", &script]).spawn().is_ok()
                });
                if !ok {
                    bail!("no terminal emulator found");
                }
            }
        }
        "open_url" => {
            Command::new(if MAC { "open" } else { "xdg-open" }).arg(&a.url).spawn()?;
        }
        "open_app" => {
            if MAC {
                Command::new("open").args(["-a", &a.app]).spawn()?;
            } else {
                Command::new(&a.app).spawn()?;
            }
        }
        "hotkey" => {
            activate(&a.target_app);
            let mut e = enigo()?;
            let keys: Vec<Key> = a.keys.iter().map(|k| key_of(k)).collect();
            for k in &keys {
                e.key(*k, Direction::Press)?;
            }
            for k in keys.iter().rev() {
                e.key(*k, Direction::Release)?;
            }
        }
        k => bail!("unknown action {k}"),
    }
    Ok(())
}

// ---- focus the window the user is looking at ---------------------------------------------------

#[cfg(target_os = "macos")]
struct WinInfo {
    pid: i64,
    app: String,
    title: String,
    x: f64,
    y: f64,
}

/// Frontmost normal window under (x, y), skipping our own windows. Quartz window bounds are global, top-left origin.
#[cfg(target_os = "macos")]
fn mac_window_at(x: f64, y: f64) -> Option<WinInfo> {
    use core_foundation::array::CFArray;
    use core_foundation::base::{CFType, TCFType};
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::number::CFNumber;
    use core_foundation::string::CFString;
    use core_graphics::window::{
        copy_window_info, kCGNullWindowID, kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly,
    };

    let me = std::process::id() as i64;
    let list: CFArray = copy_window_info(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID)?;
    for item in list.iter() {
        let dict: CFDictionary<CFString, CFType> = unsafe { CFDictionary::wrap_under_get_rule(*item as _) };
        let num = |k: &str| dict.find(CFString::new(k)).and_then(|v| v.downcast::<CFNumber>()).and_then(|n| n.to_f64());
        let text = |k: &str| dict.find(CFString::new(k)).and_then(|v| v.downcast::<CFString>()).map(|s| s.to_string());
        if num("kCGWindowLayer") != Some(0.0) || num("kCGWindowOwnerPID") == Some(me as f64) {
            continue;
        }
        let Some(b) = dict.find(CFString::new("kCGWindowBounds")) else { continue };
        let b: CFDictionary<CFString, CFType> = unsafe { CFDictionary::wrap_under_get_rule(b.as_CFTypeRef() as _) };
        let bn = |k: &str| b.find(CFString::new(k)).and_then(|v| v.downcast::<CFNumber>()).and_then(|n| n.to_f64()).unwrap_or(0.0);
        let (bx, by, bw, bh) = (bn("X"), bn("Y"), bn("Width"), bn("Height"));
        if x >= bx && x < bx + bw && y >= by && y < by + bh {
            return Some(WinInfo {
                pid: num("kCGWindowOwnerPID").unwrap_or(0.0) as i64,
                app: text("kCGWindowOwnerName").unwrap_or_default(),
                title: text("kCGWindowName").unwrap_or_default(),
                x: bx,
                y: by,
            });
        }
    }
    None
}

/// Focus the window under (x, y); optionally click there to put the caret in the field.
/// Returns a short description of the focused window.
pub fn focus_at(x: f64, y: f64, click: bool) -> String {
    #[allow(unused_mut)]
    let mut desc = String::new();
    #[cfg(target_os = "macos")]
    if let Some(w) = mac_window_at(x, y) {
        // bring the app forward and raise that exact window (AXRaise needs Accessibility)
        osascript(&format!(
            r#"tell application "System Events"
                 set p to first process whose unix id is {pid}
                 set frontmost of p to true
                 try
                   repeat with w in windows of p
                     set {{wx, wy}} to position of w
                     if (wx - {x}) ^ 2 + (wy - {y}) ^ 2 < 25 then
                       perform action "AXRaise" of w
                       exit repeat
                     end if
                   end repeat
                 end try
               end tell"#,
            pid = w.pid,
            x = w.x as i64,
            y = w.y as i64
        ));
        desc = if w.title.is_empty() { w.app } else { format!("{} — {}", w.app, w.title) };
    }
    #[cfg(not(target_os = "macos"))]
    {
        let out = sh("xdotool", &["mousemove", &(x as i64).to_string(), &(y as i64).to_string(), "getmouselocation", "--shell"]);
        if let Some(wid) = out.lines().find_map(|l| l.strip_prefix("WINDOW=")) {
            sh("xdotool", &["windowactivate", "--sync", wid]);
        }
        desc = sh("xdotool", &["getactivewindow", "getwindowname"]);
    }
    if click {
        if let Ok(mut e) = enigo() {
            let old = e.location().unwrap_or((x as i32, y as i32));
            let _ = e.move_mouse(x as i32, y as i32, Coordinate::Abs);
            std::thread::sleep(Duration::from_millis(40));
            let _ = e.button(Button::Left, Direction::Click);
            let _ = e.move_mouse(old.0, old.1, Coordinate::Abs);
        }
    }
    std::thread::sleep(Duration::from_millis(150));
    desc
}
