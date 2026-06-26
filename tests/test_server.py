"""Tests for the FastAPI HTTP server."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from language_learning.config import TutorConfig
from language_learning.controller.controller import Controller
from language_learning.models.evaluation import EvaluationResult
from language_learning.storage.memory import InMemoryStorage
from language_learning.web.server import _last_assistant, create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARMS_DIR = Path(__file__).parent.parent / "arms"


def _arms_list() -> list[dict]:
    import yaml
    with open(ARMS_DIR / "arms.yaml") as f:
        return yaml.safe_load(f)["arms"]


def _mock_llm(initiate_text="Ciao! Come stai?", response_text="Ottimo!", eval_result=None):
    from language_learning.controller.llm_client import LLMClient

    if eval_result is None:
        eval_result = EvaluationResult(
            target_attempted=True,
            target_success="partial",
            avoidance="none",
            fluency_proxy=0.7,
            novelty_proxy=0.6,
            praise="Good attempt!",
            fix_one="Use 'sono andato' not 'ho andato'.",
            next_nudge="Try using the past tense.",
        )

    client = MagicMock(spec=LLMClient)
    client.evaluate = AsyncMock(return_value=eval_result)
    client.generate_response = AsyncMock(return_value=response_text)
    client.initiate = AsyncMock(return_value=initiate_text)
    return client


@pytest.fixture
def app(tmp_path):
    """App with an in-memory controller and mocked LLM."""
    arms = _arms_list()
    storage = InMemoryStorage(arms=arms)
    cfg = TutorConfig(model="openai/gpt-4o", api_key="test")

    def factory() -> Controller:
        ctrl = Controller(language="it", config=cfg, storage=storage)
        ctrl.llm_client = _mock_llm()
        return ctrl

    return create_app(cfg, _controller_factory=factory)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def started_client(client):
    """Client with a session already started (tutor has sent first message)."""
    r = client.post("/api/start", json={"topic": "travel", "personality": "playful"})
    assert r.status_code == 200
    return client


# ---------------------------------------------------------------------------
# HTML endpoint
# ---------------------------------------------------------------------------

class TestIndex:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_content_type_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_chat_structure(self, client):
        r = client.get("/")
        body = r.text
        assert "id=\"chat\"" in body
        assert "id=\"msgInput\"" in body
        assert "id=\"sendBtn\"" in body

    def test_contains_start_modal(self, client):
        body = client.get("/").text
        assert 'id="overlay"' in body or 'id="modal"' in body

    def test_contains_api_calls(self, client):
        body = client.get("/").text
        assert "/api/start" in body
        assert "/api/turn" in body
        assert "/api/rate" in body


# ---------------------------------------------------------------------------
# /api/start
# ---------------------------------------------------------------------------

class TestApiStart:
    def test_returns_ok(self, client):
        r = client.post("/api/start", json={"topic": "travel", "personality": "playful"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True

    def test_returns_tutor_message(self, client):
        r = client.post("/api/start", json={"topic": "travel", "personality": "playful"})
        data = r.json()
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

    def test_returns_arm(self, client):
        r = client.post("/api/start", json={"topic": "free conversation", "personality": "funny"})
        data = r.json()
        assert "arm" in data
        assert isinstance(data["arm"], str)

    def test_returns_turn_count(self, client):
        r = client.post("/api/start", json={})
        data = r.json()
        assert "turn_count" in data
        assert isinstance(data["turn_count"], int)

    def test_default_topic_and_personality(self, client):
        r = client.post("/api/start", json={})
        assert r.status_code == 200

    def test_message_matches_mock(self, client):
        r = client.post("/api/start", json={"topic": "travel", "personality": "playful"})
        assert r.json()["message"] == "Ciao! Come stai?"


# ---------------------------------------------------------------------------
# /api/turn
# ---------------------------------------------------------------------------

class TestApiTurn:
    def test_returns_message(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ho mangiato la pasta."})
        assert r.status_code == 200
        assert r.json()["message"] == "Ottimo!"

    def test_returns_feedback(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ieri ho andato al cinema."})
        data = r.json()
        assert "feedback" in data
        fb = data["feedback"]
        assert "praise" in fb
        assert "fix_one" in fb
        assert "next_nudge" in fb

    def test_feedback_values(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ciao!"})
        fb = r.json()["feedback"]
        assert fb["praise"] == "Good attempt!"
        assert "andato" in fb["fix_one"]

    def test_warmup_turn_has_no_pending_rating(self, started_client):
        # First user turn after session start is a warmup — no star-rating prompt
        r = started_client.post("/api/turn", json={"text": "Parla di viaggi."})
        data = r.json()
        assert "pending_rating_turn_id" in data
        assert data["pending_rating_turn_id"] is None

    def test_returns_pending_rating_turn_id(self, started_client):
        # Second turn (after warmup) gets a rating widget
        started_client.post("/api/turn", json={"text": "Parla di viaggi."})
        r = started_client.post("/api/turn", json={"text": "Mi piace tanto."})
        data = r.json()
        assert "pending_rating_turn_id" in data
        assert data["pending_rating_turn_id"] is not None

    def test_returns_arm(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ciao!"})
        assert "arm" in r.json()

    def test_returns_turn_count(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ciao!"})
        assert r.json()["turn_count"] >= 1

    def test_empty_text_returns_400(self, started_client):
        r = started_client.post("/api/turn", json={"text": ""})
        assert r.status_code == 400

    def test_whitespace_only_returns_400(self, started_client):
        r = started_client.post("/api/turn", json={"text": "   "})
        assert r.status_code == 400

    def test_multiple_turns(self, started_client):
        for text in ["Prima cosa.", "Seconda cosa.", "Terza cosa."]:
            r = started_client.post("/api/turn", json={"text": text})
            assert r.status_code == 200
        assert r.json()["turn_count"] >= 3

    def test_works_without_prior_start(self, client):
        r = client.post("/api/turn", json={"text": "Ciao!"})
        assert r.status_code == 200

    def test_cefr_field_present(self, started_client):
        r = started_client.post("/api/turn", json={"text": "Ciao!"})
        assert "cefr" in r.json()


# ---------------------------------------------------------------------------
# /api/rate
# ---------------------------------------------------------------------------

class TestApiRate:
    def _get_turn_id(self, started_client):
        # First turn is warmup (no rating); second has pending_rating_turn_id
        started_client.post("/api/turn", json={"text": "Ciao!"})
        r = started_client.post("/api/turn", json={"text": "Come stai?"})
        return r.json()["pending_rating_turn_id"]

    def test_rate_5_stars(self, started_client):
        tid = self._get_turn_id(started_client)
        r = started_client.post("/api/rate", json={"turn_id": tid, "stars": 5})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_rate_1_star(self, started_client):
        tid = self._get_turn_id(started_client)
        r = started_client.post("/api/rate", json={"turn_id": tid, "stars": 1})
        assert r.status_code == 200

    def test_rate_3_stars(self, started_client):
        tid = self._get_turn_id(started_client)
        r = started_client.post("/api/rate", json={"turn_id": tid, "stars": 3})
        assert r.status_code == 200

    def test_invalid_stars_0(self, started_client):
        r = started_client.post("/api/rate", json={"turn_id": "any", "stars": 0})
        assert r.status_code == 400

    def test_invalid_stars_6(self, started_client):
        r = started_client.post("/api/rate", json={"turn_id": "any", "stars": 6})
        assert r.status_code == 400

    def test_unknown_turn_id_is_no_op(self, started_client):
        r = started_client.post("/api/rate", json={"turn_id": "nonexistent", "stars": 3})
        assert r.status_code == 200

    def test_rate_clears_pending(self, started_client):
        tid = self._get_turn_id(started_client)
        started_client.post("/api/rate", json={"turn_id": tid, "stars": 4})
        r2 = started_client.post("/api/turn", json={"text": "Secondo turno."})
        assert r2.status_code == 200


# ---------------------------------------------------------------------------
# /api/state
# ---------------------------------------------------------------------------

class TestApiState:
    def test_returns_language(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        assert r.json()["language"] == "it"

    def test_returns_session_active(self, client):
        r = client.get("/api/state")
        assert "session_active" in r.json()

    def test_session_active_after_start(self, started_client):
        r = started_client.get("/api/state")
        assert r.json()["session_active"] is True

    def test_returns_stats(self, client):
        r = client.get("/api/state")
        assert "stats" in r.json()
        assert isinstance(r.json()["stats"], str)

    def test_returns_topic_info(self, started_client):
        r = started_client.get("/api/state")
        assert "topic_info" in r.json()
        assert "travel" in r.json()["topic_info"]

    def test_returns_why(self, client):
        r = client.get("/api/state")
        assert "why" in r.json()

    def test_returns_cefr(self, started_client):
        started_client.post("/api/turn", json={"text": "Ciao!"})
        r = started_client.get("/api/state")
        assert "cefr" in r.json()

    def test_turn_count_increments(self, started_client):
        started_client.post("/api/turn", json={"text": "Uno."})
        started_client.post("/api/turn", json={"text": "Due."})
        r = started_client.get("/api/state")
        assert r.json()["turn_count"] >= 2


# ---------------------------------------------------------------------------
# /api/compact
# ---------------------------------------------------------------------------

class TestApiEnd:
    def test_end_session_returns_ok(self, started_client):
        r = started_client.post("/api/end")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_session_inactive_after_end(self, started_client):
        started_client.post("/api/end")
        r = started_client.get("/api/state")
        assert r.json()["session_active"] is False

    def test_end_without_session_is_harmless(self, client):
        r = client.post("/api/end")
        assert r.status_code == 200


class TestApiCompact:
    def test_returns_ok_status(self, started_client):
        r = started_client.post("/api/compact")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_returns_path(self, started_client):
        r = started_client.post("/api/compact")
        assert "path" in r.json()


# ---------------------------------------------------------------------------
# _last_assistant helper
# ---------------------------------------------------------------------------

class TestLastAssistant:
    def test_empty_state(self):
        from language_learning.models.state import AppState
        assert _last_assistant(AppState(language="it")) == ""

    def test_returns_last_assistant_message(self):
        from language_learning.models.state import AppState, ChatMessage
        state = AppState(
            language="it",
            chat_messages=[
                ChatMessage(role="user", text="Ciao"),
                ChatMessage(role="assistant", text="Ciao a te!"),
            ],
        )
        assert _last_assistant(state) == "Ciao a te!"

    def test_returns_most_recent(self):
        from language_learning.models.state import AppState, ChatMessage
        state = AppState(
            language="it",
            chat_messages=[
                ChatMessage(role="assistant", text="First"),
                ChatMessage(role="user", text="Reply"),
                ChatMessage(role="assistant", text="Second"),
            ],
        )
        assert _last_assistant(state) == "Second"

    def test_only_user_messages_returns_empty(self):
        from language_learning.models.state import AppState, ChatMessage
        state = AppState(
            language="it",
            chat_messages=[ChatMessage(role="user", text="Hello")],
        )
        assert _last_assistant(state) == ""


# ---------------------------------------------------------------------------
# /test endpoint
# ---------------------------------------------------------------------------

class TestConnectionTest:
    def _mock_completion(self, text="OK"):
        from unittest.mock import AsyncMock, MagicMock
        resp = MagicMock()
        resp.choices[0].message.content = text
        return AsyncMock(return_value=resp)

    def test_returns_ok_status(self, client, monkeypatch):
        monkeypatch.setattr("litellm.acompletion", self._mock_completion())
        r = client.get("/test")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_returns_model(self, client, monkeypatch):
        monkeypatch.setattr("litellm.acompletion", self._mock_completion())
        r = client.get("/test")
        assert r.json()["model"] == "openai/gpt-4o"

    def test_returns_response_text(self, client, monkeypatch):
        monkeypatch.setattr("litellm.acompletion", self._mock_completion("  OK  "))
        r = client.get("/test")
        assert r.json()["response"] == "OK"

    def test_returns_latency_ms(self, client, monkeypatch):
        monkeypatch.setattr("litellm.acompletion", self._mock_completion())
        r = client.get("/test")
        assert "latency_ms" in r.json()
        assert isinstance(r.json()["latency_ms"], int)

    def test_returns_error_on_llm_failure(self, client, monkeypatch):
        from unittest.mock import AsyncMock
        monkeypatch.setattr(
            "litellm.acompletion",
            AsyncMock(side_effect=Exception("connection refused")),
        )
        r = client.get("/test")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "error"
        assert "connection refused" in data["error"]
        assert "latency_ms" in data

    def test_does_not_require_active_session(self, client, monkeypatch):
        monkeypatch.setattr("litellm.acompletion", self._mock_completion())
        r = client.get("/test")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_full_turn_cycle(self, client):
        r1 = client.post("/api/start", json={"topic": "daily life", "personality": "direct"})
        assert r1.status_code == 200
        assert r1.json()["message"]

        # First turn is warmup — no rating expected
        r2 = client.post("/api/turn", json={"text": "Sono andato a fare la spesa."})
        assert r2.status_code == 200
        assert r2.json()["pending_rating_turn_id"] is None

        # Second turn gets a rating widget
        r3 = client.post("/api/turn", json={"text": "Ho anche cucinato la pasta."})
        assert r3.status_code == 200
        turn_id = r3.json()["pending_rating_turn_id"]
        assert turn_id

        r4 = client.post("/api/rate", json={"turn_id": turn_id, "stars": 4})
        assert r4.status_code == 200

        r5 = client.get("/api/state")
        assert r5.json()["turn_count"] >= 2

    def test_multiple_sessions_sequential(self, client):
        for topic in ["travel", "sports"]:
            r = client.post("/api/start", json={"topic": topic, "personality": "funny"})
            assert r.status_code == 200
