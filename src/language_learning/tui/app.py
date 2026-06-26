"""Textual TUI application for the language tutor."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from language_learning.config import TutorConfig
from language_learning.controller.controller import Controller
from language_learning.models.session_context import (
    PERSONALITIES,
    TOPICS,
    parse_start_command,
)
from language_learning.models.state import AppState, ChatMessage
from language_learning.tui.widgets import DebugPanel, FeedbackRail, StarRating

CSS_PATH = Path(__file__).parent / "css" / "tutor.tcss"


class TutorApp(App):
    """Adaptive conversational language tutor TUI."""

    CSS_PATH = CSS_PATH
    TITLE = "Language Tutor"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+d", "toggle_debug", "Debug"),
    ]

    def __init__(
        self,
        language: str = "it",
        data_dir: str = ".",
        config: TutorConfig | None = None,
    ) -> None:
        super().__init__()
        self.language = language
        self.data_dir = data_dir
        self._config = config
        self.controller: Controller | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-panes"):
            with VerticalScroll(id="conversation"):
                yield Static(
                    "Type /start <topic> [as <personality>] to begin a conversation.\n"
                    f"Topics: {', '.join(TOPICS)}\n"
                    f"Personalities: {', '.join(PERSONALITIES.keys())}",
                    id="welcome-msg",
                )
            yield FeedbackRail(id="feedback-rail")
        yield DebugPanel(id="debug-panel")
        yield StarRating(id="star-rating")
        yield Input(placeholder="Type in your target language...", id="user-input")
        yield Footer()

    async def on_mount(self) -> None:
        if self._config is None:
            from language_learning.config import load_config
            self._config = load_config()

        self.controller = Controller(
            language=self.language,
            config=self._config,
            data_dir=self.data_dir,
            on_state_change=self._on_state_change,
        )
        await self.controller.initialize()
        self._update_subtitle()
        self.query_one("#user-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.clear()

        if text.startswith("/"):
            await self._handle_command(text)
            return

        if not self.controller.session_context:
            self._add_system_message(
                "No session active. Use /start <topic> [as <personality>] to begin."
            )
            return

        # Auto-rate pending turn if the user just types a new message
        if self.controller.app_state.pending_rating_turn_id:
            self.controller.submit_engagement_rating(
                self.controller.app_state.pending_rating_turn_id, stars=3
            )
            self.query_one("#star-rating", StarRating).remove_class("visible")

        self.run_worker(self.controller.process_turn(text), exclusive=True)

    def on_star_rating_star_rated(self, event: StarRating.StarRated) -> None:
        turn_id = self.controller.app_state.pending_rating_turn_id
        if turn_id:
            self.controller.submit_engagement_rating(turn_id, event.stars)
            self.query_one("#star-rating", StarRating).remove_class("visible")
            self.query_one("#user-input", Input).focus()

    async def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/start":
            await self._handle_start(arg)
        elif cmd == "/stop":
            self.controller.end_session()
            self.controller.session_context = None
            self._add_system_message("Session ended.")
        elif cmd == "/topic":
            self._add_system_message(self.controller.get_topic_info())
        elif cmd == "/newsession":
            self.controller.end_session()
            await self.controller.initialize()
            self._add_system_message(
                "Session reset. Use /start <topic> [as <personality>] to begin."
            )
        elif cmd == "/language":
            if arg in ("it", "es"):
                await self.controller.switch_language(arg)
                self._add_system_message(f"Switched to {arg.upper()}.")
            else:
                self._add_system_message("Usage: /language it|es")
        elif cmd == "/stats":
            self._add_system_message(self.controller.get_stats())
        elif cmd == "/arm":
            arm = self.controller.current_arm
            if arm:
                self._add_system_message(f"Current arm: {arm.arm_id}\nIntent: {arm.intent}")
            else:
                self._add_system_message("No arm selected.")
        elif cmd == "/why":
            self._add_system_message(self.controller.get_why())
        elif cmd == "/cefr":
            self._add_system_message(self.controller.get_cefr_summary())
        elif cmd == "/debug":
            self.action_toggle_debug()
        elif cmd == "/export":
            self._add_system_message("Transcripts are auto-saved to logs/<lang>/sessions/")
        elif cmd == "/compact":
            self.controller.storage.compact_logs(self.controller.language)
            self._add_system_message("Logs compacted.")
        elif cmd in ("/quit", "/exit"):
            await self.action_quit()
        else:
            self._add_system_message(
                f"Unknown command: {cmd}\n"
                "Available: /start, /stop, /topic, /newsession, /language, "
                "/stats, /arm, /why, /cefr, /debug, /export, /compact, /quit"
            )

    async def _handle_start(self, arg: str) -> None:
        if not arg:
            self._add_system_message(
                "Usage: /start <topic> [as <personality>]\n"
                f"Topics: {', '.join(TOPICS)}\n"
                f"Personalities: {', '.join(PERSONALITIES.keys())}\n"
                "Examples:\n"
                "  /start current news as intellectual\n"
                "  /start travel as playful\n"
                "  /start daily life"
            )
            return

        topic, personality, error = parse_start_command(arg)
        if error:
            self._add_system_message(error)
            return

        self._add_system_message(f"Starting conversation: {topic} (as {personality})...")
        self.run_worker(
            self.controller.start_session(topic, personality), exclusive=True
        )

    def _add_system_message(self, text: str) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        conv.mount(Static(f"[italic]{text}[/italic]", classes="message-system"))
        conv.scroll_end(animate=False)

    def _on_state_change(self, state: AppState) -> None:
        self.call_later(self._apply_state, state)

    def _update_subtitle(self) -> None:
        if not self.controller:
            return
        state = self.controller.app_state
        parts = [f"{state.language.upper()}", f"arm: {state.current_arm}", f"turn {state.turn_count}"]
        if self.controller.session_context:
            parts.insert(1, f"topic: {self.controller.session_context.topic}")
        self.sub_title = " | ".join(parts)
        if state.status_line:
            self.sub_title += f" | {state.status_line}"

    def _apply_state(self, state: AppState) -> None:
        conv = self.query_one("#conversation", VerticalScroll)

        for w in self.query("#welcome-msg"):
            w.remove()

        existing = conv.query(".message-user, .message-assistant")
        existing_count = len(existing)

        if existing_count > len(state.chat_messages):
            for widget in existing:
                widget.remove()
            existing_count = 0

        for msg in state.chat_messages[existing_count:]:
            prefix = "You" if msg.role == "user" else "Tutor"
            conv.mount(Static(f"[bold]{prefix}:[/bold] {msg.text}", classes=f"message-{msg.role}"))
        conv.scroll_end(animate=False)

        self.query_one("#feedback-rail", FeedbackRail).update_feedback(state.feedback_panel)

        star_widget = self.query_one("#star-rating", StarRating)
        if state.pending_rating_turn_id:
            star_widget.add_class("visible")
        else:
            star_widget.remove_class("visible")

        self._update_subtitle()
        self.query_one("#debug-panel", DebugPanel).update_debug(state.debug_state)

    def action_toggle_debug(self) -> None:
        self.query_one("#debug-panel", DebugPanel).toggle_class("visible")

    async def action_quit(self) -> None:
        if self.controller:
            self.controller.end_session()
            self.controller.generate_reports()
            await self.controller.shutdown()
        self.exit()
