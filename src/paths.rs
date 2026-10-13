//! XDG locations (also used on macOS, like most CLI tools).

use std::path::PathBuf;

fn xdg(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(default))
        .join("precog")
}

/// ~/.config/precog: settings
pub fn config_dir() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config")
}

/// ~/.local/share/precog: trained gaze model, speech models
pub fn data_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share")
}

/// ~/.cache/precog: debug screenshots
pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache")
}

pub fn gaze_model() -> PathBuf {
    data_dir().join("gaze_model.json")
}

/// ONNX models: $PRECOG_MODELS, else ./models/onnx next to the binary's repo, else the data dir.
pub fn onnx_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PRECOG_MODELS") {
        return PathBuf::from(p);
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/onnx");
    if repo.exists() {
        return repo;
    }
    data_dir().join("onnx")
}
