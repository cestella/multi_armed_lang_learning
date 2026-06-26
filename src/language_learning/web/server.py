"""FastAPI HTTP server — wraps Controller with an iMessage-style chat frontend."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

import litellm
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from language_learning.config import TutorConfig
from language_learning.controller.controller import Controller
from language_learning.models.state import AppState


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

CEFR_LEVELS = ["A1", "A1+", "A2", "A2+", "A2+/B1-", "B1-", "B1", "B1+", "B1+/B2-", "B2"]


class StartRequest(BaseModel):
    topic: str = "free conversation"
    personality: str = "encouraging"
    cefr_override: str | None = None  # None means "sync with empirical"


class TurnRequest(BaseModel):
    text: str


class RateRequest(BaseModel):
    turn_id: str
    stars: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    config: TutorConfig,
    language: str = "it",
    data_dir: str = ".",
    _controller_factory: Callable[[], Controller] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Pass *_controller_factory* in tests to inject a pre-built (mocked)
    Controller instead of constructing one from *config*/*language*/*data_dir*.
    """
    controller: Controller | None = None
    _latest_state: list[AppState] = []

    def _on_state_change(state: AppState) -> None:
        _latest_state.clear()
        _latest_state.append(state.model_copy(deep=True))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal controller
        if _controller_factory is not None:
            controller = _controller_factory()
        else:
            controller = Controller(
                language=language,
                config=config,
                data_dir=os.path.abspath(data_dir),
            )
        controller.on_state_change = _on_state_change
        await controller.initialize()
        yield
        await controller.shutdown()

    app = FastAPI(title="Language Tutor", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Connection test (no session required)
    # ------------------------------------------------------------------

    @app.get("/test")
    async def test_connection() -> dict[str, Any]:
        """Ping the configured LLM and report latency + response."""
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": "Reply with only the word: OK"}],
            "max_tokens": 10,
            "timeout": config.timeout,
        }
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.api_key:
            kwargs["api_key"] = config.api_key

        t0 = time.monotonic()
        try:
            resp = await litellm.acompletion(**kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)
            text = resp.choices[0].message.content or ""
            return {
                "status": "ok",
                "model": config.model,
                "response": text.strip(),
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "status": "error",
                "model": config.model,
                "error": str(exc),
                "latency_ms": latency_ms,
            }

    # ------------------------------------------------------------------
    # HTML frontend
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_CHAT_HTML)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.post("/api/start")
    async def start(req: StartRequest) -> dict[str, Any]:
        override = req.cefr_override if req.cefr_override in CEFR_LEVELS else None
        _latest_state.clear()
        await controller.start_session(req.topic, req.personality, cefr_override=override)
        state = _latest_state[-1] if _latest_state else controller.app_state
        return {
            "ok": True,
            "message": _last_assistant(state),
            "arm": state.current_arm,
            "turn_count": state.turn_count,
        }

    @app.post("/api/turn")
    async def turn(req: TurnRequest) -> dict[str, Any]:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        _latest_state.clear()
        await controller.process_turn(req.text)
        state = _latest_state[-1] if _latest_state else controller.app_state
        fp = state.feedback_panel
        return {
            "message": _last_assistant(state),
            "feedback": {
                "praise": fp.praise,
                "fix_one": fp.fix_one,
                "micro_rule": fp.micro_rule,
                "next_nudge": fp.next_nudge,
                "hint_phrase": fp.hint_phrase,
                "retry_prompt": fp.retry_prompt,
            },
            "pending_rating_turn_id": state.pending_rating_turn_id,
            "arm": state.current_arm,
            "turn_count": state.turn_count,
            "cefr": controller.get_cefr_summary() if controller.cefr_state else "",
        }

    @app.post("/api/rate")
    async def rate(req: RateRequest) -> dict[str, Any]:
        if not 1 <= req.stars <= 5:
            raise HTTPException(status_code=400, detail="stars must be 1–5")
        controller.submit_engagement_rating(req.turn_id, req.stars)
        return {"ok": True}

    @app.get("/api/state")
    async def api_state() -> dict[str, Any]:
        s = controller.app_state
        empirical = getattr(controller.cefr_state, "overall_estimate", None) if controller.cefr_state else None
        return {
            "language": controller.language,
            "arm": s.current_arm,
            "turn_count": s.turn_count,
            "session_active": s.session_active,
            "stats": controller.get_stats(),
            "topic_info": controller.get_topic_info(),
            "cefr": controller.get_cefr_summary() if controller.cefr_state else "",
            "cefr_level": empirical,
            "cefr_override": controller.cefr_override,
            "why": controller.get_why(),
        }

    @app.post("/api/end")
    async def end_session() -> dict[str, Any]:
        controller.end_session()
        return {"ok": True}

    @app.post("/api/compact")
    async def compact() -> dict[str, Any]:
        return controller.storage.compact_logs(controller.language)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_assistant(state: AppState) -> str:
    for msg in reversed(state.chat_messages):
        if msg.role == "assistant":
            return msg.text
    return ""


# ---------------------------------------------------------------------------
# iMessage-style chat HTML (single file, no build step)
# ---------------------------------------------------------------------------

_CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Language Tutor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --blue: #0b93f6;
  --blue-hover: #0a7fd4;
  --grey-bubble: #e5e5ea;
  --grey-text: #1c1c1e;
  --bg: #f2f2f7;
  --surface: #ffffff;
  --border: #d1d1d6;
  --muted: #8e8e93;
  --feedback-bg: #eef6ff;
  --feedback-border: #b8d8f8;
  --star: #ffd60a;
  --danger: #ff3b30;
  --success: #34c759;
  --r-bubble: 18px;
  --r-sm: 10px;
  --r-md: 14px;
  font-size: 16px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --grey-bubble: #3a3a3c;
    --grey-text: #f2f2f7;
    --bg: #1c1c1e;
    --surface: #2c2c2e;
    --border: #3a3a3c;
    --feedback-bg: #1a2d45;
    --feedback-border: #3a6090;
  }
}

body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  color: var(--grey-text);
}

/* ── Header ──────────────────────────────────────────── */
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  backdrop-filter: saturate(180%) blur(20px);
}
.avatar {
  width: 38px; height: 38px; border-radius: 50%;
  background: linear-gradient(135deg, #0b93f6 0%, #5e5ce6 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
}
.header-info { flex: 1; min-width: 0; }
.header-name { font-size: 15px; font-weight: 600; }
.header-sub  { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }
.header-btns { display: flex; gap: 4px; }
.hbtn {
  background: none; border: none; cursor: pointer;
  font-size: 13px; color: var(--muted);
  padding: 5px 8px; border-radius: 8px;
  transition: background 0.15s, color 0.15s;
}
.hbtn:hover { background: var(--grey-bubble); color: var(--grey-text); }

/* ── Chat ──────────────────────────────────────────────── */
#chat {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 2px;
  scroll-behavior: smooth;
}

.msg-row { display: flex; flex-direction: column; margin: 3px 0; }
.msg-row.user      { align-items: flex-end; }
.msg-row.assistant { align-items: flex-start; }

.bubble {
  max-width: min(72%, 480px);
  padding: 9px 14px;
  font-size: 15px; line-height: 1.45;
  word-break: break-word;
}
.msg-row.user .bubble {
  background: var(--blue); color: #fff;
  border-radius: var(--r-bubble) var(--r-bubble) 4px var(--r-bubble);
}
.msg-row.assistant .bubble {
  background: var(--grey-bubble); color: var(--grey-text);
  border-radius: var(--r-bubble) var(--r-bubble) var(--r-bubble) 4px;
}

/* feedback card */
.fb-card {
  max-width: min(72%, 480px);
  width: min(72%, 480px);
  margin-top: 6px; padding: 8px 12px 10px;
  background: var(--feedback-bg);
  border: 1px solid var(--feedback-border);
  border-left: 3px solid var(--blue);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  font-size: 13px; line-height: 1.5;
}
.fb-header {
  font-size: 10px; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted);
  margin-bottom: 6px;
}
.fb-row { margin: 2px 0; }
.fb-lbl { font-weight: 600; margin-right: 4px; }
.fb-lbl.ok    { color: var(--success); }
.fb-lbl.fix   { color: var(--danger); }
.fb-lbl.nudge { color: var(--blue); }

/* star rating */
.star-row { display: flex; align-items: center; gap: 6px; margin-top: 6px; }
.star-hint { font-size: 11px; color: var(--muted); }
.stars { display: flex; gap: 1px; }
.star {
  font-size: 19px; cursor: pointer;
  color: var(--border); transition: color 0.1s, transform 0.1s;
  user-select: none; line-height: 1;
}
.star:hover, .star.lit { color: var(--star); }
.star:hover { transform: scale(1.2); }
.star.locked { cursor: default; pointer-events: none; }

/* typing indicator */
.typing-row { display: flex; margin: 3px 0; }
.typing-bubble {
  background: var(--grey-bubble);
  border-radius: var(--r-bubble) var(--r-bubble) var(--r-bubble) 4px;
  padding: 10px 14px; display: flex; align-items: center; gap: 4px;
}
.dot {
  width: 6px; height: 6px; background: var(--muted);
  border-radius: 50%; animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
  0%,60%,100% { transform: translateY(0); }
  30%         { transform: translateY(-5px); }
}

/* ── Input bar ──────────────────────────────────────────── */
#inputBar {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 8px 12px;
  display: flex; align-items: flex-end; gap: 8px;
  flex-shrink: 0;
}
#msgInput {
  flex: 1; resize: none;
  border: 1px solid var(--border); border-radius: 20px;
  padding: 8px 14px; font-size: 15px; font-family: inherit;
  background: var(--bg); color: var(--grey-text);
  outline: none; max-height: 120px; overflow-y: auto;
  line-height: 1.4; transition: border-color 0.15s;
}
#msgInput:focus { border-color: var(--blue); }
#sendBtn {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--blue); border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; transition: background 0.15s;
}
#sendBtn:disabled { background: var(--border); cursor: default; }
#sendBtn svg { fill: #fff; width: 15px; height: 15px; }

/* ── Start modal ──────────────────────────────────────────── */
#overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,.5);
  display: flex; align-items: center; justify-content: center;
  z-index: 100;
}
#modal {
  background: var(--surface); border-radius: 20px;
  padding: 28px 24px; width: min(360px, 92vw);
  box-shadow: 0 12px 48px rgba(0,0,0,.25);
}
#modal h2 { font-size: 19px; font-weight: 600; text-align: center; margin-bottom: 20px; }
.fgroup { margin-bottom: 14px; }
.fgroup label { display: block; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 5px; }
.fgroup select {
  width: 100%; padding: 9px 12px; border-radius: var(--r-sm);
  border: 1px solid var(--border); background: var(--bg);
  color: var(--grey-text); font-size: 14px; font-family: inherit;
  outline: none;
}
#startBtn {
  width: 100%; padding: 11px;
  background: var(--blue); color: #fff; border: none;
  border-radius: var(--r-md); font-size: 15px; font-weight: 600;
  cursor: pointer; margin-top: 4px; transition: background 0.15s;
}
#startBtn:hover { background: var(--blue-hover); }
#startBtn:disabled { background: var(--border); cursor: default; }

/* ── Info sheet ──────────────────────────────────────────── */
#sheet-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.4); z-index: 50;
  align-items: flex-end; justify-content: center;
}
#sheet-overlay.open { display: flex; }
#sheet {
  background: var(--surface); border-radius: 20px 20px 0 0;
  padding: 20px 20px 36px; width: 100%; max-width: 520px;
  max-height: 65vh; overflow-y: auto;
  animation: slideup .25s ease;
}
@keyframes slideup { from { transform: translateY(100%); } to { transform: translateY(0); } }
.sheet-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.sheet-header h3 { font-size: 16px; font-weight: 600; }
.sheet-close { background: none; border: none; font-size: 20px; cursor: pointer; color: var(--muted); }
#sheetPre { font-family: "SF Mono", ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; background: var(--bg); padding: 12px; border-radius: var(--r-sm); line-height: 1.6; }
</style>
</head>
<body>

<!-- Start modal -->
<div id="overlay">
  <div id="modal">
    <h2>🌍 Language Tutor</h2>
    <div class="fgroup">
      <label>Topic</label>
      <select id="topicSel">
        <option value="free conversation">Free conversation</option>
        <option value="a philosophical or moral dilemma to debate">Philosophy &amp; moral dilemmas</option>
        <option value="current news">Current news</option>
        <option value="travel">Travel</option>
        <option value="technology">Technology</option>
        <option value="culture">Culture</option>
        <option value="politics">Politics</option>
        <option value="sports">Sports</option>
        <option value="daily life">Daily life</option>
        <option value="food and cooking">Food &amp; cooking</option>
        <option value="history">History</option>
      </select>
    </div>
    <div class="fgroup">
      <label>Tutor personality</label>
      <select id="persSelect">
        <option value="encouraging">Encouraging</option>
        <option value="funny">Funny</option>
        <option value="patient">Patient</option>
        <option value="intellectual">Intellectual</option>
        <option value="direct">Direct</option>
        <option value="playful">Playful</option>
        <option value="formal">Formal</option>
      </select>
    </div>
    <div class="fgroup">
      <label>Your level</label>
      <select id="cefrSelect">
        <option value="">Sync with empirical (…)</option>
        <option value="A1">A1 — Beginner</option>
        <option value="A1+">A1+ — High Beginner</option>
        <option value="A2">A2 — Elementary</option>
        <option value="A2+">A2+ — High Elementary</option>
        <option value="A2+/B1-">A2+/B1- — Lower Intermediate</option>
        <option value="B1-">B1- — Pre-Intermediate</option>
        <option value="B1">B1 — Intermediate</option>
        <option value="B1+">B1+ — Upper Intermediate</option>
        <option value="B1+/B2-">B1+/B2- — Lower Advanced</option>
        <option value="B2">B2 — Advanced</option>
      </select>
    </div>
    <button id="startBtn" onclick="startSession()">Begin session</button>
  </div>
</div>

<!-- Info sheet -->
<div id="sheet-overlay" onclick="closeSheet(event)">
  <div id="sheet">
    <div class="sheet-header">
      <h3 id="sheetTitle"></h3>
      <button class="sheet-close" onclick="closeSheetDirect()">✕</button>
    </div>
    <pre id="sheetPre"></pre>
  </div>
</div>

<!-- Help dialog -->
<div id="helpOverlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:200;align-items:center;justify-content:center;" onclick="closeHelp(event)">
  <div id="helpBox" style="background:var(--surface);border-radius:20px;padding:28px 24px;width:min(480px,92vw);max-height:80vh;overflow-y:auto;box-shadow:0 12px 48px rgba(0,0,0,.25);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <strong style="font-size:17px;">How this works</strong>
      <button onclick="document.getElementById('helpOverlay').style.display='none'" style="background:none;border:none;font-size:20px;cursor:pointer;color:var(--muted)">✕</button>
    </div>
    <div style="font-size:14px;line-height:1.65;color:var(--grey-text);">
      <p><strong>Goal:</strong> Language acquisition through natural conversation — not drills. You chat in your target language; the tutor steers the conversation toward the grammar patterns you struggle with, without breaking the flow.</p>
      <br>
      <p><strong>Multi-armed bandit (UCB1):</strong> The system has six "arms" — conversational intents like <em>narrative_yesterday</em> (past-tense narration) or <em>hypothetical_future</em> (conditionals). After each turn it balances <em>exploration</em> (trying all arms) with <em>exploitation</em> (favouring arms that have produced good learning signal). The 📊 button shows current arm scores.</p>
      <br>
      <p><strong>Reward signal:</strong> Each turn produces a learning reward (from the LLM's evaluation of your grammar, fluency, and novelty) blended 60/40 with your star rating. Your stars capture how enjoyable the conversation felt — the tutor optimises for both learning and engagement.</p>
      <br>
      <p><strong>Skill tracking:</strong> The tutor tracks mastery of specific structures (past narration, verb agreement, conditionals…). When you stumble on something, it locks onto that scene for a few turns to give you more practice, then moves on once you nail it twice in a row.</p>
      <br>
      <p><strong>CEFR level:</strong> The system estimates your level (A1–B2) from the evaluation signals and adjusts vocabulary and sentence complexity accordingly. You can override this in the session-start dialog if the estimate seems off.</p>
      <br>
      <p><strong>Feedback card:</strong> After each message you send, a blue card appears below the tutor's reply showing one correction (✎), a brief grammar rule (📌), and a nudge for the next turn (→). It also has a star rating widget. This never appears in the conversation itself — the tutor won't break character.</p>
      <br>
      <p><strong>Star rating:</strong> Rate after each reply. 5 = great conversation, 1 = boring or too hard. This directly shapes which conversational direction the tutor takes next.</p>
    </div>
  </div>
</div>

<!-- Header -->
<header>
  <div class="avatar">🤖</div>
  <div class="header-info">
    <div class="header-name" id="hName">Language Tutor</div>
    <div class="header-sub"  id="hSub">Choose a topic to begin</div>
  </div>
  <div class="header-btns">
    <button class="hbtn" onclick="showSheet('stats')" title="Bandit stats">📊</button>
    <button class="hbtn" onclick="showSheet('why')"   title="Why this arm?">🤔</button>
    <button class="hbtn" onclick="showSheet('cefr')"  title="CEFR level">📈</button>
    <button class="hbtn" onclick="showHelp()"         title="How this works">?</button>
    <button class="hbtn" id="endBtn" onclick="endSession()" title="End session" style="display:none">✕ End</button>
  </div>
</header>

<!-- Chat -->
<div id="chat"></div>

<!-- Input -->
<div id="inputBar">
  <textarea id="msgInput" rows="1" placeholder="Type in your target language…"
    disabled oninput="resize(this)" onkeydown="onKey(event)"></textarea>
  <button id="sendBtn" disabled onclick="send()">
    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
  </button>
</div>

<script>
'use strict';

let busy = false;

/* ── Populate empirical CEFR label when modal opens ──────────────────────────────────────────── */
async function populateCefrLabel() {
  try {
    const s = await get('/api/state');
    const level = s.cefr_level || '?';
    document.querySelector('#cefrSelect option[value=""]').textContent =
      `Sync with empirical (${level})`;
  } catch (_) {}
}
populateCefrLabel();

/* ── Session start ──────────────────────────────────────────── */
async function startSession() {
  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  const cefrVal = document.getElementById('cefrSelect').value;
  try {
    const r = await post('/api/start', {
      topic: document.getElementById('topicSel').value,
      personality: document.getElementById('persSelect').value,
      cefr_override: cefrVal || null,
    });
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('endBtn').style.display = '';
    setEnabled(true);
    updateHeader(r.arm, document.getElementById('topicSel').value + ' · ' + document.getElementById('persSelect').value);
    if (r.message) addBubble('assistant', r.message, null, null);
    scrollEnd();
  } catch (e) {
    alert('Could not start: ' + e.message);
    btn.disabled = false;
    btn.textContent = 'Begin session';
  }
}

/* ── Send a message ──────────────────────────────────────────── */
async function send() {
  const el = document.getElementById('msgInput');
  const text = el.value.trim();
  if (!text || busy) return;
  busy = true;
  setEnabled(false);
  el.value = ''; resize(el);
  addBubble('user', text, null, null);
  scrollEnd();
  showTyping();
  try {
    const r = await post('/api/turn', { text });
    hideTyping();
    addBubble('assistant', r.message, r.feedback, r.pending_rating_turn_id);
    updateHeader(r.arm, null);
    scrollEnd();
  } catch (e) {
    hideTyping();
    addBubble('assistant', '⚠️ ' + e.message, null, null);
  } finally {
    busy = false;
    setEnabled(true);
    el.focus();
  }
}

/* ── End session / new session ──────────────────────────────────────────── */
async function endSession() {
  if (!confirm('End this session and start a new one?')) return;
  try { await post('/api/end', {}); } catch (_) {}
  document.getElementById('chat').innerHTML = '';
  document.getElementById('hName').textContent = 'Language Tutor';
  document.getElementById('hSub').textContent = 'Choose a topic to begin';
  document.getElementById('endBtn').style.display = 'none';
  document.getElementById('startBtn').disabled = false;
  document.getElementById('startBtn').textContent = 'Begin session';
  document.getElementById('cefrSelect').value = '';
  setEnabled(false);
  document.getElementById('overlay').style.display = 'flex';
  populateCefrLabel();
}

/* ── Rating ──────────────────────────────────────────── */
async function rate(turnId, stars, starsEl) {
  try {
    await post('/api/rate', { turn_id: turnId, stars });
    starsEl.querySelectorAll('.star').forEach((s, i) => {
      s.classList.toggle('lit', i < stars);
      s.classList.add('locked');
    });
  } catch (_) {}
}

/* ── DOM helpers ──────────────────────────────────────────── */
function addBubble(role, text, feedback, turnId) {
  const chat = document.getElementById('chat');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  if (turnId) row.dataset.turn = turnId;

  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  row.appendChild(b);

  if (role === 'assistant') {
    const card = makeFbCard(feedback, turnId);
    if (card) row.appendChild(card);
  }
  chat.appendChild(row);
}

function makeFbCard(fb, turnId) {
  const rows = [];
  if (fb?.praise)      rows.push({ cls: 'ok',    icon: '✓', text: fb.praise });
  if (fb?.fix_one)     rows.push({ cls: 'fix',   icon: '✎', text: fb.fix_one });
  if (fb?.micro_rule)  rows.push({ cls: '',      icon: '📌', text: fb.micro_rule });
  if (fb?.next_nudge)  rows.push({ cls: 'nudge', icon: '→', text: fb.next_nudge });
  if (fb?.hint_phrase) rows.push({ cls: '',      icon: '💡', text: fb.hint_phrase });
  if (fb?.retry_prompt)rows.push({ cls: '',      icon: '↺', text: fb.retry_prompt });

  if (!rows.length && !turnId) return null;

  const card = document.createElement('div');
  card.className = 'fb-card';

  if (rows.length) {
    const hdr = document.createElement('div');
    hdr.className = 'fb-header';
    hdr.textContent = 'Coach';
    card.appendChild(hdr);
  }

  rows.forEach(({ cls, icon, text }) => {
    const d = document.createElement('div');
    d.className = 'fb-row';
    d.innerHTML = `<span class="fb-lbl ${cls}">${icon}</span>${escHtml(text)}`;
    card.appendChild(d);
  });

  if (turnId) {
    const sr = document.createElement('div');
    sr.className = 'star-row';
    sr.innerHTML = '<span class="star-hint">Rate conversation:</span>';
    const stars = document.createElement('div');
    stars.className = 'stars';
    for (let i = 1; i <= 5; i++) {
      const s = document.createElement('span');
      s.className = 'star'; s.textContent = '★'; s.dataset.v = i;
      s.addEventListener('mouseover', () => litUpTo(stars, i));
      s.addEventListener('mouseout',  () => litUpTo(stars, 0));
      s.addEventListener('click',     () => rate(turnId, i, stars));
      stars.appendChild(s);
    }
    sr.appendChild(stars);
    card.appendChild(sr);
  }
  return card;
}

function litUpTo(container, n) {
  container.querySelectorAll('.star').forEach((s, i) => s.classList.toggle('lit', i < n));
}

let typingEl = null;
function showTyping() {
  if (typingEl) return;
  typingEl = document.createElement('div');
  typingEl.className = 'typing-row';
  typingEl.innerHTML = '<div class="typing-bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  document.getElementById('chat').appendChild(typingEl);
  scrollEnd();
}
function hideTyping() { typingEl?.remove(); typingEl = null; }

function scrollEnd() {
  const c = document.getElementById('chat');
  c.scrollTop = c.scrollHeight;
}

function setEnabled(on) {
  document.getElementById('msgInput').disabled = !on;
  document.getElementById('sendBtn').disabled  = !on;
}

function updateHeader(arm, sub) {
  if (arm) document.getElementById('hName').textContent = 'Language Tutor · ' + arm.replace(/_/g,' ');
  if (sub) document.getElementById('hSub').textContent = sub;
}

function resize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── Info sheet ──────────────────────────────────────────── */
async function showSheet(kind) {
  try {
    const s = await get('/api/state');
    const map = { stats: ['Bandit stats', s.stats], why: ['Why this arm?', s.why], cefr: ['CEFR estimate', s.cefr || 'No data yet.'] };
    const [title, text] = map[kind];
    document.getElementById('sheetTitle').textContent = title;
    document.getElementById('sheetPre').textContent = text;
    document.getElementById('sheet-overlay').classList.add('open');
  } catch (e) { alert('Could not load state: ' + e.message); }
}
function closeSheet(e) {
  if (e.target === document.getElementById('sheet-overlay')) closeSheetDirect();
}
function closeSheetDirect() {
  document.getElementById('sheet-overlay').classList.remove('open');
}

/* ── Help dialog ──────────────────────────────────────────── */
function showHelp() {
  document.getElementById('helpOverlay').style.display = 'flex';
}
function closeHelp(e) {
  if (e.target === document.getElementById('helpOverlay'))
    document.getElementById('helpOverlay').style.display = 'none';
}

/* ── Fetch helpers ──────────────────────────────────────────── */
async function post(path, body) {
  const r = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || r.statusText); }
  return r.json();
}
async function get(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}
</script>
</body>
</html>
"""
