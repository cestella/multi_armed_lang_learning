"""Custom Textual widgets for the tutor TUI."""

from __future__ import annotations

from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from language_learning.models.state import DebugState, FeedbackPanel


class FeedbackRail(VerticalScroll):
    """Right-side feedback panel showing per-message coaching."""

    def compose(self):
        yield Static("Feedback", classes="feedback-label", id="fb-title")
        yield Static("", id="fb-praise")
        yield Static("", id="fb-fix")
        yield Static("", id="fb-rule")
        yield Static("", id="fb-nudge")
        yield Static("", id="fb-hint")
        yield Static("", id="fb-retry")

    def update_feedback(self, panel: FeedbackPanel) -> None:
        self.query_one("#fb-praise", Static).update(
            f"[green]Praise:[/green] {panel.praise}" if panel.praise else ""
        )
        self.query_one("#fb-fix", Static).update(
            f"[yellow]Fix:[/yellow] {panel.fix_one}" if panel.fix_one else ""
        )
        self.query_one("#fb-rule", Static).update(
            f"[cyan]Rule:[/cyan] {panel.micro_rule}" if panel.micro_rule else ""
        )
        self.query_one("#fb-nudge", Static).update(
            f"[blue]Next:[/blue] {panel.next_nudge}" if panel.next_nudge else ""
        )
        self.query_one("#fb-hint", Static).update(
            f"[magenta]Pattern:[/magenta] {panel.hint_phrase}" if panel.hint_phrase else ""
        )
        self.query_one("#fb-retry", Static).update(
            f"[bold blue]Retry:[/bold blue] {panel.retry_prompt}" if panel.retry_prompt else ""
        )


class DebugPanel(VerticalScroll):
    """Optional debug panel showing bandit state."""

    def compose(self):
        yield Static("Debug Info", classes="feedback-label")
        yield Static("", id="debug-content")

    def update_debug(self, state: DebugState) -> None:
        lines = [f"Arm: {state.current_arm}"]
        if state.last_reward is not None:
            lines.append(f"Reward: {state.last_reward:.4f}")
            for k, v in state.reward_components.items():
                lines.append(f"  {k}: {v:.4f}")
        if state.focus_skill:
            lines.append(f"Focus: {state.focus_skill} ({state.focus_turns} turns)")
        if state.arm_scores:
            lines.append("Scores:")
            for aid in sorted(state.arm_scores.keys()):
                lines.append(f"  {aid}: {state.arm_scores[aid]:.4f}")
        self.query_one("#debug-content", Static).update("\n".join(lines))


class StarRating(Horizontal):
    """Star rating widget (1-5) for engagement feedback."""

    class StarRated(Message):
        """Posted when user selects a star rating."""

        def __init__(self, stars: int) -> None:
            self.stars = stars
            super().__init__()

    BINDINGS = [
        ("1", "rate(1)", "1 star"),
        ("2", "rate(2)", "2 stars"),
        ("3", "rate(3)", "3 stars"),
        ("4", "rate(4)", "4 stars"),
        ("5", "rate(5)", "5 stars"),
    ]

    def compose(self):
        yield Static("Rate this question: ", id="star-label")
        for i in range(1, 6):
            yield Button(str(i), id=f"star-{i}", classes="star-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        star_num = int(event.button.id.split("-")[1])
        self.post_message(self.StarRated(star_num))

    def action_rate(self, stars: int) -> None:
        self.post_message(self.StarRated(stars))
