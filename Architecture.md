# Adaptive Conversational Language Tutor
## Architecture Specification (Italian + Spanish, No DB, MCP + TUI)

Version: v0.2  
Status: Updated design (dual-language support)

---

## 1. Overview

This project implements an **adaptive conversational language tutor** that runs in a **terminal user interface (TUI)**.

The tutor conducts a natural conversation with the user while continuously adapting based on learning signals extracted from the user’s messages. Adaptation uses a **multi-armed bandit (UCB1)** to choose conversational “pressures” (arms) that maximize learning gain.

Key requirements:

- **Natural conversation** (no explicit “quiz mode” prompts)
- **Per-message feedback** shown separately from the chat transcript
- **Adaptive arm selection** via UCB1 + cooldown
- **No database** (all state persisted to disk as JSON/JSONL/Markdown)
- **Crash-safe / resumable** via event-sourcing
- **Supports both Italian and Spanish** with clean separation of logs and state per language
- **MCP servers** provide storage/bandit/evaluation; the **Controller** owns the loop

---

## 2. System Architecture

Components:

```
User
 │
 ▼
TUI (Textual)
 │
 ▼
Controller (event-sourced orchestration)
 │
 ├── Storage MCP   (disk I/O only)
 ├── Bandit MCP    (UCB1 select + update)
 └── Evaluator MCP (LLM evaluation -> JSON)
```

### Responsibilities

| Component | Responsibility |
|---|---|
| TUI | Render split-pane UI, capture input, emit UI events |
| Controller | Turn orchestration, retries/timeouts, idempotency, updates AppState |
| Storage MCP | Append events, read/write JSON snapshots, append markdown |
| Bandit MCP | Deterministic UCB1 + cooldown, persisted bandit state |
| Evaluator MCP | Evaluate user message and return strict JSON (feedback + signals) |

**Rule:** The TUI never calls MCP directly.  
All tool interactions go: `TUI → Controller → MCP → Controller → TUI`.

---

## 3. Dual-Language Support Model

### 3.1 Principle

Italian and Spanish are both supported by treating **language as a first-class partition** for:

- state snapshots
- event logs
- transcripts
- compaction outputs

This avoids complex filtering during replay and keeps per-language learner models and bandit stats clean.

### 3.2 One language per active session

A session is associated with exactly one target language:

- `it` (Italian)
- `es` (Spanish)

The user can switch languages via a command (see §11). Switching languages **ends the current session** and starts/continues another language session.

---

## 4. File Layout (No DB)

All persistent artifacts are files on disk.

Recommended layout:

```
state/
  it/
    learner_profile.json
    bandit_state.json
  es/
    learner_profile.json
    bandit_state.json

logs/
  it/
    events.jsonl
    sessions/
      YYYY-MM-DD.md
    compacted/
      <timestamp>_summary.json
  es/
    events.jsonl
    sessions/
      YYYY-MM-DD.md
    compacted/
      <timestamp>_summary.json

arms/
  arms.yaml               # language-agnostic arms definition
  structure_hints_it.json # optional, shallow hints
  structure_hints_es.json # optional, shallow hints

scripts/
  cleanup_mcp.py

README.md
ARCHITECTURE.md
```

### Notes

- **No database** allowed (no sqlite, postgres, redis, etc.).
- Events are append-only JSONL.
- Snapshots are derived caches (can be rebuilt from events).
- Writes must be **atomic**: write temp → rename.

---

## 5. UI Design (Textual TUI)

Layout:

```
┌─────────────────────────────────────────────┬───────────────┐
│                                             │               │
│                                             │   Feedback    │
│              Conversation                   │               │
│                                             │  ✅ praise     │
│                                             │  🔧 fix_one    │
│                                             │  🧠 micro_rule │
│                                             │  🎯 next_nudge │
│                                             │               │
├─────────────────────────────────────────────┴───────────────┤
│                     User Input Box                          │
└─────────────────────────────────────────────────────────────┘
```

### UI behavior

- Chat transcript is natural and flowing.
- Feedback rail shows **per-message coaching** (short, actionable).
- Optional debug toggle shows arm + reward info.

---

## 6. Controller Design (Message Passing + Event Sourcing)

### 6.1 Core pattern

The Controller is an event-driven orchestrator:

- Receives UI events from the TUI
- Appends **domain events** to `events.jsonl` (source of truth)
- Executes bounded side effects via MCP calls
- Updates in-memory `AppState`
- Triggers UI re-render

### 6.2 UI events (in-memory only)

- `SubmitText(text: str)`
- `Command(cmd: str)`
- `ToggleDebug(on: bool)`
- `QuitRequested()`

TUI emits into an async queue owned by the Controller.

### 6.3 Rendering contract (Controller → TUI)

TUI renders from a single `AppState`:

- `chat_messages[]`
- `feedback_panel` (praise/fix_one/micro_rule/next_nudge)
- `status_line` (e.g. “Evaluating…”, tool failure notices)
- `debug_state` (optional)

---

## 7. Event Log (Source of Truth)

Events stored in:

- `logs/it/events.jsonl`
- `logs/es/events.jsonl`

Each line is a JSON object:

```json
{
  "event_id": "uuid",
  "turn_id": "uuid-or-null",
  "ts": "ISO8601",
  "type": "user_submitted",
  "payload": {
    "language": "it",
    "text": "Sono andato al ristorante ieri"
  }
}
```

### Required event fields

- `event_id`: UUID
- `turn_id`: UUID (null for session-level events)
- `ts`: ISO8601 timestamp
- `type`: event name
- `payload`: JSON data

### Required event types

- `session_started`
- `user_submitted`
- `evaluation_completed`
- `reward_computed`
- `bandit_updated`
- `arm_selected`
- `assistant_responded`
- `tool_failed` (any MCP failure)
- `session_ended` (optional but recommended)

**Rule:** Every event payload MUST include `"language": "it" | "es"` for safety.

---

## 8. Snapshot State Files

Snapshots are derived caches. They can be rebuilt from event logs.

### 8.1 learner_profile.json (per language)

Path:
- `state/it/learner_profile.json`
- `state/es/learner_profile.json`

Example:

```json
{
  "schema_version": 1,
  "language": "it",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "preferences": {},
  "recurring_wins": [
    { "label": "clear meaning", "count": 3 }
  ],
  "recurring_fixes": [
    { "label": "past tense mismatch", "count": 5 }
  ],
  "recent_focus": {
    "label": "past tense narration",
    "since": "ISO8601"
  }
}
```

### 8.2 bandit_state.json (per language)

Path:
- `state/it/bandit_state.json`
- `state/es/bandit_state.json`

Schema is prescriptive in §9.

---

## 9. Multi-Armed Bandit (UCB1, Prescriptive)

### 9.1 Bandit state schema

```json
{
  "schema_version": 1,
  "algo": "ucb1",
  "c": 0.7,
  "cooldown_max_repeat": 2,
  "cooldown_penalty": 0.15,
  "total_pulls": 0,
  "arms": {
    "narrative_yesterday": { "n": 0, "mean": 0.0 }
  },
  "recent_arms": ["narrative_yesterday"]
}
```

### 9.2 Selection algorithm

1. **Warm start exploration**
   - If any arm has `n == 0`, select one of them deterministically by `arm_id` lexical order.

2. **UCB1 score**
   - For each arm with `n > 0` compute:

```
base_ucb = mean + c * sqrt( ln(total_pulls) / n )
cooldown_count = occurrences of arm_id in recent_arms
score = base_ucb - cooldown_penalty * cooldown_count
```

3. **Pick best**
   - Choose highest score; tie-break by arm_id lexical sort.

4. **Repeat limit**
   - If the last `cooldown_max_repeat` arms in `recent_arms` are identical, force pick the best-scoring different arm (deterministic ties).

### 9.3 Update algorithm

On each turn:

- `total_pulls += 1`
- `n += 1`
- `mean = mean + (reward - mean) / n`
- append arm_id to `recent_arms`, trim to length `cooldown_max_repeat`

### 9.4 Idempotency

Bandit updates must be idempotent by `turn_id`.  
If a `bandit_updated` event already exists for that `turn_id`, do not apply again during resume/replay.

---

## 10. Arms System (Language-Agnostic)

Arms are **conversation pressures**, not grammar drills.

Stored in `arms/arms.yaml` and shared by both languages.

Each arm includes:

- `arm_id`
- `intent` (private)
- `prompt_templates` (2–4)
- `fallback_nudges` (2–3)
- optional `tags` (e.g. narrative, instructions)

Arms should be general enough to work in both Italian and Spanish.

---

## 11. Commands (In-App)

Commands are implemented by the tutor TUI (not Codex).

Examples:

- `/newsession` — start a new session in the current language
- `/language it` — switch to Italian (end current session, load Italian snapshots/logs)
- `/language es` — switch to Spanish
- `/stats` — show bandit stats
- `/arm` — show current arm
- `/export` — export transcript for current session
- `/compact` — compact logs for current language
- `/debug on|off` — toggle debug panel
- `/why` — show last reward components + arm selection detail

---

## 12. Evaluator (Structured JSON Output)

Evaluator MCP returns strict JSON:

- `praise` (<=120 chars)
- `fix_one` (<=160 chars) — exactly one focus issue
- `micro_rule` (<=120 chars or null)
- `recast` (1–2 sentences, natural correction)
- `next_nudge` (a question to continue conversation)
- `target_attempted` (bool)
- `target_success` ("no" | "partial" | "yes")
- `errors[]` small list of `{type, note}`
- `avoidance` ("none" | "weak" | "strong")
- `fluency_proxy` (0..1)
- `novelty_proxy` (0..1)

Evaluator input MUST include `language`.

Optional: provide shallow language hints from:
- `arms/structure_hints_it.json`
- `arms/structure_hints_es.json`

These hints are not rule engines; they only help standardize labels.

---

## 13. Reward Function

Reward is computed locally by the Controller (pure function), clamped to [0, 1]:

```
reward =
  0.35*(target_attempted?1:0) +
  0.25*(target_success=="yes"?1:target_success=="partial"?0.5:0) +
  0.20*novelty_proxy +
  0.10*fluency_proxy -
  0.30*(avoidance=="strong"?1:avoidance=="weak"?0.5:0)
```

Reward computation is logged as `reward_computed` event.

---

## 14. Controller Turn Workflow (Prescriptive)

Given a `SubmitText(text)`:

1. Create `turn_id`
2. Append `user_submitted` event
3. Update UI chat transcript with user message immediately
4. Set status: “Evaluating…”

5. Call Evaluator MCP (`timeout=15s`, retry up to 2)
   - On success: append `evaluation_completed`
   - Update feedback rail
   - On failure: append `tool_failed` and produce a graceful assistant reply without bandit update

6. Compute reward locally
   - append `reward_computed`

7. Call Bandit MCP update (`timeout=5s`, retries)
   - append `bandit_updated`

8. Call Bandit MCP select (`timeout=5s`, retries)
   - append `arm_selected`

9. Generate assistant response (v1: deterministic composition)
   - conversational reply includes evaluator `recast` subtly
   - follow-up question uses evaluator `next_nudge`
   - append `assistant_responded`

10. Update UI: add assistant message; clear status

---

## 15. Reliability

### 15.1 Timeouts + retries

All MCP calls are bounded:

- evaluator: 15s, retry 2
- bandit: 5s, retry 2
- storage: 5s, retry 2

On repeated failure, emit `tool_failed` and keep the UI responsive.

### 15.2 Atomic writes

All snapshot JSON writes must be atomic:

1. write `file.tmp`
2. fsync
3. rename to target filename

### 15.3 Resume

On startup:

1. determine active language (default `it`, or last used)
2. load snapshots for that language
3. replay events if needed
4. detect incomplete last turn
5. resume idempotently

---

## 16. Transcripts (Derived)

For each language:

- `logs/<lang>/sessions/YYYY-MM-DD.md`

These are human-readable and can be rebuilt from events.

---

## 17. Compaction (Per Language)

Compaction is per-language and does not conflict with Codex.

Command:

- `/compact`

Output:

- `logs/<lang>/compacted/<timestamp>_summary.json`

Summary includes:

- bandit snapshot
- top recurring fixes/wins
- recent focus label
- tail excerpt of recent turns (ids + short text)
- schema_version

Compaction does not delete original `events.jsonl` automatically.

Optional future: allow `--use-compacted` to replay summary + tail events, but full replay from original JSONL must always work.

---

## 18. Design Principles

- **Natural conversation first**: feedback is separate from chat
- **Adaptive probing**: by steering conversation, not quizzing
- **Deterministic core**: UCB1, lexical tie-breaks, idempotent replay
- **No DB**: everything inspectable and portable as files
- **Debuggable**: every decision is recorded in events

---

## 19. Summary

This architecture provides:

- adaptive conversational tutoring
- dual-language separation (Italian + Spanish)
- deterministic and inspectable learning loop
- crash-safe operation without a database
- MCP-based modularity for future expansion
