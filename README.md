# Adaptive Conversational Language Tutor

An adaptive language tutor that conducts natural conversations in Italian or Spanish while continuously adapting to the learner's weaknesses. It uses a **multi-armed bandit (UCB1)** to select conversational pressures and **LiteLLM** to route to any LLM provider.

The core idea: language acquisition happens through conversation, not drills. The tutor steers the conversation toward grammatical structures the learner struggles with, but the learner only experiences a natural exchange.

## Quick start

### 1. Install

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd multi_armed_lang_learning
uv sync
```

### 2. Configure

Copy the example config and set your LLM:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to point at your provider:

```yaml
# Local ds4 (DeepSeek V4 via antirez/ds4)
model: openai/deepseek-v4
api_base: http://10.0.4.105:8000/v1
api_key: not-needed

# --- OR ---

# Anthropic API
model: anthropic/claude-sonnet-4-6
# set ANTHROPIC_API_KEY env var

# --- OR ---

# OpenAI
model: openai/gpt-4o
# set OPENAI_API_KEY env var
```

### 3. Run

**Web UI (iMessage-style chat):**

```bash
uv run language-tutor serve --language it --data-dir ./lang_data
# Open http://127.0.0.1:8080
```

**Terminal UI (Textual):**

```bash
uv run language-tutor tui --language it --data-dir ./lang_data
```

Both interfaces share the same controller and storage layer.

## CLI reference

```
language-tutor serve [OPTIONS]    HTTP chat server (web UI)
language-tutor tui   [OPTIONS]    Textual terminal UI

Options (both commands):
  --language   it|es          Target language (default: it)
  --data-dir   PATH           State and log directory (default: .)
  --config     PATH           Config file (default: auto-discover)

Options (serve only):
  --host       HOST           Bind host (default: 127.0.0.1)
  --port       PORT           Bind port (default: 8080)
```

Config auto-discovery searches: `./config.yaml`, then `~/.config/language-tutor/config.yaml`.

## Web UI

Open `http://127.0.0.1:8080` after running `serve`. You'll see a session-start dialog — pick a topic and personality, then chat in your target language.

- **Blue bubbles** = tutor messages
- **Grey area below** = grammar feedback (correction, rule, nudge)
- **Stars** = rate conversation engagement (1–5); updates the bandit
- **Header icons**: 📊 bandit stats · 🤔 arm reasoning · 📈 CEFR estimate

## TUI commands

| Command | Description |
|---------|-------------|
| `/start <topic> [as <personality>]` | Begin a session |
| `/stop` | End the current session |
| `/language it\|es` | Switch target language |
| `/stats` | Bandit arm statistics |
| `/arm` | Current arm and intent |
| `/why` | Last reward breakdown |
| `/compact` | Compact event log |

Topics: `current news`, `travel`, `technology`, `culture`, `politics`, `sports`, `daily life`, `free conversation`

Personalities: `encouraging`, `funny`, `patient`, `intellectual`, `direct`, `playful`, `formal`

## How it works

### Turn loop

1. Learner types in the target language.
2. The LLM **evaluates** the message: detects errors, classifies them, measures fluency/novelty.
3. A **reward** is computed from the evaluation; the bandit update is deferred until the learner rates engagement.
4. The LLM **responds** naturally in the target language, nudging toward the current arm's grammar focus.
5. After the learner rates (1–5 stars), the **bandit updates** and selects the next arm.

### Arms (conversational pressures)

Six arms defined in `arms/arms.yaml`. Each is a conversational *intent* that naturally elicits specific grammar:

| Arm | Intent | Skills targeted |
|-----|--------|-----------------|
| `description_scene` | Describe a place or person | vocabulary, agreement |
| `emotion_reaction` | Express feelings | vocabulary, past narration |
| `hypothetical_future` | Conditionals and planning | conditional/future |
| `instruction_explain` | Procedural language | vocabulary, prepositions |
| `narrative_yesterday` | Past-tense narration | past narration |
| `opinion_compare` | Comparisons and opinions | vocabulary, conditional |

### Adaptation

- **UCB1 bandit** balances exploration vs exploitation; skill-based priors bias toward the learner's weakest areas.
- **Skill tracker** maintains mastery, confidence, and scaffold need per skill, updating each turn.
- **Conversation focus** locks the tutor into the current scene for 2–4 turns when the learner struggles, then releases on two consecutive successes.
- **Reward blending** mixes learning signal (60%) with engagement rating (40%).
- **CEFR estimation** continuously updates an A1–B2 estimate per domain.

## Data layout

```
lang_data/
  arms/
    arms.yaml                   # arm definitions (shared)
  state/
    it/
      bandit_state.json         # UCB1 stats
      learner_profile.json      # wins and recurring fixes
      skill_state.json          # per-skill mastery
      cefr_state.json           # CEFR estimate
      conversation_focus.json   # active focus lock
      skill_history.jsonl       # skill trend log
      cefr_history.jsonl        # CEFR trend log
      session_metrics.jsonl     # per-turn attempt/avoidance data
  logs/
    it/
      events.jsonl              # append-only event log (source of truth)
      sessions/
        2026-06-20.md           # human-readable transcript
      compacted/
        <ts>_summary.json
  reports/
    it/
      progress_summary.md
      cefr_progress.png
      skill_mastery.png
      scaffold_trend.png
      attempt_avoidance.png
```

## Architecture

```
Web browser / TUI
       │
       ▼
  HTTP server (FastAPI)  ──OR──  Textual TUI
       │                               │
       └───────────────┬───────────────┘
                       ▼
                  Controller
            (pure Python, UI-agnostic)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      LLMClient    StorageBackend  Core logic
     (LiteLLM)   (filesystem /    (bandit, skills,
                   in-memory)      reward, CEFR)
```

The **Controller** owns all domain state and fires an `on_state_change(AppState)` callback after each mutation. The HTTP server and TUI are thin adapters that call `process_turn()`, `submit_engagement_rating()`, etc.

The **StorageBackend** interface lets you swap filesystem for any other persistence layer. `InMemoryStorage` is used in tests for speed and isolation.

## Tests

```bash
uv run pytest tests/ -q
```

396 tests covering: bandit, reward, skills, CEFR, storage (parametrized over filesystem and in-memory), controller integration, HTTP server endpoints, config loading, and progress reports.

## Package structure

```
src/language_learning/
  cli/main.py               # Click CLI (tui, serve)
  config.py                 # TutorConfig + load_config()
  controller/
    controller.py           # Event-driven orchestrator
    llm_client.py           # LiteLLM wrapper
  core/
    bandit.py               # UCB1 select + update (pure)
    cefr.py                 # CEFR estimation
    focus.py                # Conversation focus lifecycle
    reward.py               # Reward computation and blending
    skills.py               # Skill mastery updates
    storage.py              # Atomic file primitives
    compression.py          # Conversation compression (future)
  models/                   # Pydantic models
  reporting/
    progress_reports.py     # Matplotlib progress charts
  storage/
    base.py                 # StorageBackend ABC
    filesystem.py           # FilesystemStorage
    memory.py               # InMemoryStorage
  tui/
    app.py                  # Textual TUI
    widgets.py              # FeedbackRail, StarRating, etc.
  web/
    server.py               # FastAPI app + HTML frontend
```
