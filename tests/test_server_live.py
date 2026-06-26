"""Integration test: real LLM call through /test endpoint against ds4.

Run with:
    uv run pytest tests/test_server_live.py -m integration -v -s

Requires:
    1. A config.yaml pointing at a live LLM endpoint (auto-discovered).
    2. macOS: grant your terminal app Local Network access in
       System Settings → Privacy & Security → Local Network.
       Without it, Homebrew/venv Python can't reach LAN IPs even though
       curl and system binaries can.
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from language_learning.config import load_config
from language_learning.web.server import create_app


def _reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _llm_host_port() -> tuple[str, int]:
    """Parse host and port from config api_base, falling back to OpenAI."""
    try:
        cfg = load_config()
        if cfg.api_base:
            from urllib.parse import urlparse
            p = urlparse(cfg.api_base)
            return p.hostname or "api.openai.com", p.port or 443
    except Exception:
        pass
    return "api.openai.com", 443


_host, _port = _llm_host_port()
_reason = f"LLM endpoint {_host}:{_port} not reachable (check network / macOS Local Network privacy)"
requires_llm = pytest.mark.skipif(not _reachable(_host, _port), reason=_reason)


@pytest.fixture(scope="module")
def live_client():
    config = load_config()
    app = create_app(config, language="it", data_dir="/tmp/lang_data_test")
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
@requires_llm
def test_live_llm_connection(live_client):
    r = live_client.get("/test")
    assert r.status_code == 200
    data = r.json()
    print(f"\n/test response: {data}")
    assert data["status"] == "ok", f"LLM error: {data.get('error')}"
    assert data["response"]
    assert data["latency_ms"] >= 0
