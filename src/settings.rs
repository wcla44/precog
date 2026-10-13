//! Persistent settings: ~/.config/precog/config.json (mode 600: it may hold an API key).
//! Same file and keys as the Python version.

use crate::paths;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Ask the LLM for suggestions.
    pub agent: bool,
    /// Show the gaze ring.
    pub dot: bool,
    /// Hand gestures (pinch / fist to accept, open hand to dictate, swipe to cancel).
    pub gesture: bool,
    /// Learn from mouse clicks.
    pub learn: bool,
    /// Fixation seconds before asking the LLM.
    pub dwell: f64,
    /// Minimum seconds between LLM calls.
    pub interval: f64,
    /// Open hand = voice typing into the window you look at.
    pub voice: bool,
    /// Whisper model: tiny | base | small | large-v3-turbo
    pub stt_model: String,
    /// "" = auto-detect, else "en", "fr", ...
    pub stt_language: String,
    /// Also click where you look (puts the caret in that field; gaze is ~2-3 cm accurate).
    pub click_focus: bool,
    /// Flip swipe direction (mirrored camera).
    pub mirror: bool,
    /// auto | anthropic | claude-cli | openai
    pub provider: String,
    /// Empty = provider default.
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub api_key_env: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            agent: true,
            dot: true,
            gesture: true,
            learn: true,
            dwell: 1.8,
            interval: 12.0,
            voice: true,
            stt_model: "base".into(),
            stt_language: String::new(),
            click_focus: false,
            mirror: false,
            provider: "auto".into(),
            model: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            api_key_env: String::new(),
        }
    }
}

pub fn path() -> PathBuf {
    paths::config_dir().join("config.json")
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read(path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        use std::io::Write;
        let p = path();
        std::fs::create_dir_all(p.parent().unwrap())?;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&p)?;
        f.write_all(&serde_json::to_vec_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }
}
