"""LLM providers: one vision call, text in / text out."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile

import httpx

# provider -> label, default model, model presets, default base URL, API key env var
PROVIDERS = {
    "auto": dict(label="Auto (API key → else Claude CLI)", default="", presets=[], base_url="", key_env=""),
    "anthropic": dict(
        label="Anthropic API",
        default="claude-sonnet-5-5",
        presets=["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"],
        base_url="", key_env="ANTHROPIC_API_KEY",
    ),
    "claude-cli": dict(
        label="Claude Code CLI (your login)",
        default="sonnet",
        presets=["sonnet", "opus", "haiku"],
        base_url="", key_env="",
    ),
    "openai": dict(
        label="OpenAI-compatible",
        default="",
        presets=[],
        base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
    ),
}

# handy OpenAI-compatible endpoints (model must accept images)
ENDPOINTS = [
    ("OpenAI", "https://api.openai.com/v1"),
    ("Ollama (local)", "http://localhost:11434/v1"),
    ("OVHcloud AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
]


def resolve(cfg: dict) -> tuple[str, str]:
    """-> (provider, model) actually used."""
    provider = cfg.get("provider", "auto")
    if provider == "auto":
        provider = "anthropic" if (cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")) else "claude-cli"
    model = cfg.get("model") or PROVIDERS[provider]["default"]
    return provider, model


def _api_key(cfg: dict, provider: str) -> str:
    env = cfg.get("api_key_env") or PROVIDERS[provider]["key_env"]
    return cfg.get("api_key") or (os.environ.get(env, "") if env else "")


def complete(cfg: dict, jpeg: bytes, prompt: str, timeout: float = 120) -> str:
    provider, model = resolve(cfg)
    b64 = base64.b64encode(jpeg).decode()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=_api_key(cfg, provider) or None, base_url=cfg.get("base_url") or None)
        msg = client.messages.create(
            model=model, max_tokens=600, timeout=timeout,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    if provider == "openai":
        if not model:
            raise RuntimeError("OpenAI-compatible provider: set a model (tray → Model → Custom…)")
        base = (cfg.get("base_url") or PROVIDERS["openai"]["base_url"]).rstrip("/")
        key = _api_key(cfg, provider)
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json={"model": model, "max_tokens": 600, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    if provider == "claude-cli":
        if not shutil.which("claude"):
            raise RuntimeError("`claude` CLI not found: install Claude Code or pick another provider")
        d = tempfile.mkdtemp(prefix="precog-")
        path = os.path.join(d, "screen.jpg")
        with open(path, "wb") as f:
            f.write(jpeg)
        out = subprocess.run(
            ["claude", "-p", f"Read the image {path} then follow these instructions.\n\n{prompt}",
             "--output-format", "json", "--model", model, "--allowedTools", "Read"],
            capture_output=True, text=True, timeout=timeout, cwd=d,
        )
        if out.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {(out.stderr or out.stdout)[-200:]}")
        return json.loads(out.stdout)["result"]

    raise RuntimeError(f"unknown provider {provider}")
