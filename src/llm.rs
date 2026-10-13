//! LLM providers: one vision call (screenshot + prompt) -> text.

use crate::settings::Settings;
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};
use std::time::Duration;

pub struct Provider {
    pub id: &'static str,
    pub label: &'static str,
    pub default_model: &'static str,
    pub presets: &'static [&'static str],
    pub base_url: &'static str,
    pub key_env: &'static str,
}

pub const PROVIDERS: &[Provider] = &[
    Provider { id: "auto", label: "Auto (API key, else Claude CLI)", default_model: "", presets: &[], base_url: "", key_env: "" },
    Provider {
        id: "anthropic",
        label: "Anthropic API",
        default_model: "claude-sonnet-5-5",
        presets: &["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"],
        base_url: "https://api.anthropic.com",
        key_env: "ANTHROPIC_API_KEY",
    },
    Provider {
        id: "claude-cli",
        label: "Claude Code CLI (your login)",
        default_model: "sonnet",
        presets: &["sonnet", "opus", "haiku"],
        base_url: "",
        key_env: "",
    },
    Provider {
        id: "openai",
        label: "OpenAI-compatible",
        default_model: "",
        presets: &[],
        base_url: "https://api.openai.com/v1",
        key_env: "OPENAI_API_KEY",
    },
];

/// Handy OpenAI-compatible endpoints (the model must accept images).
pub const ENDPOINTS: &[(&str, &str)] = &[
    ("OpenAI", "https://api.openai.com/v1"),
    ("Ollama (local)", "http://localhost:11434/v1"),
    ("OVHcloud AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
];

pub fn provider(id: &str) -> &'static Provider {
    PROVIDERS.iter().find(|p| p.id == id).unwrap_or(&PROVIDERS[0])
}

/// -> (provider, model) actually used.
pub fn resolve(cfg: &Settings) -> (&'static Provider, String) {
    let mut p = provider(&cfg.provider);
    if p.id == "auto" {
        let has_key = !cfg.api_key.is_empty() || std::env::var("ANTHROPIC_API_KEY").is_ok_and(|k| !k.is_empty());
        p = provider(if has_key { "anthropic" } else { "claude-cli" });
    }
    let model = if cfg.model.is_empty() { p.default_model.to_string() } else { cfg.model.clone() };
    (p, model)
}

fn api_key(cfg: &Settings, p: &Provider) -> String {
    if !cfg.api_key.is_empty() {
        return cfg.api_key.clone();
    }
    let env = if cfg.api_key_env.is_empty() { p.key_env } else { cfg.api_key_env.as_str() };
    if env.is_empty() {
        String::new()
    } else {
        std::env::var(env).unwrap_or_default()
    }
}

pub fn complete(cfg: &Settings, jpeg: &[u8], prompt: &str) -> Result<String> {
    let (p, model) = resolve(cfg);
    let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg);
    let http = || reqwest::blocking::Client::builder().timeout(Duration::from_secs(120)).build();
    match p.id {
        "anthropic" => {
            let base = if cfg.base_url.is_empty() { p.base_url } else { cfg.base_url.trim_end_matches('/') };
            let resp: Value = http()?
                .post(format!("{base}/v1/messages"))
                .header("x-api-key", api_key(cfg, p))
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": model, "max_tokens": 600,
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt},
                    ]}],
                }))
                .send()?
                .error_for_status()?
                .json()?;
            Ok(resp["content"]
                .as_array()
                .context("unexpected Anthropic response")?
                .iter()
                .filter_map(|b| b["text"].as_str())
                .collect())
        }
        "openai" => {
            if model.is_empty() {
                bail!("OpenAI-compatible provider: set a model in Settings");
            }
            let base = if cfg.base_url.is_empty() { p.base_url } else { cfg.base_url.trim_end_matches('/') };
            let key = api_key(cfg, p);
            let mut req = http()?.post(format!("{base}/chat/completions"));
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
            let resp: Value = req
                .json(&json!({
                    "model": model, "max_tokens": 600,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}},
                        {"type": "text", "text": prompt},
                    ]}],
                }))
                .send()?
                .error_for_status()?
                .json()?;
            resp["choices"][0]["message"]["content"].as_str().map(str::to_string).context("unexpected response")
        }
        "claude-cli" => {
            let dir = std::env::temp_dir().join(format!("precog-{}", std::process::id()));
            std::fs::create_dir_all(&dir)?;
            let img = dir.join("screen.jpg");
            std::fs::write(&img, jpeg)?;
            let out = std::process::Command::new("claude")
                .current_dir(&dir)
                .args(["-p", &format!("Read the image {} then follow these instructions.\n\n{prompt}", img.display())])
                .args(["--output-format", "json", "--model", &model, "--allowedTools", "Read"])
                .output()
                .map_err(|e| anyhow!("`claude` CLI not available ({e}): install Claude Code or pick another provider"))?;
            if !out.status.success() {
                bail!("claude CLI failed: {}", String::from_utf8_lossy(&out.stderr).chars().take(200).collect::<String>());
            }
            let v: Value = serde_json::from_slice(&out.stdout)?;
            v["result"].as_str().map(str::to_string).context("no result from claude CLI")
        }
        other => bail!("unknown provider {other}"),
    }
}
