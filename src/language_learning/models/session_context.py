"""Session context model — topic and personality for a conversation session."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel


TOPICS = [
    "current news",
    "travel",
    "technology",
    "culture",
    "politics",
    "sports",
    "daily life",
    "free conversation",
]

PERSONALITIES = {
    "encouraging": "Supportive tone, positive reinforcement, celebrate small wins",
    "funny": "Light humor, jokes, playful wordplay in the target language",
    "patient": "Slow pace, repeat key phrases, never rush the learner",
    "intellectual": "Thoughtful questions, deeper follow-ups, explore nuance",
    "direct": "Concise feedback, no filler, straight to the point",
    "playful": "Humor and light tone, casual style, emojis welcome",
    "formal": "Polished register, formal address forms (Lei/usted), business-like",
}


class SessionContext(BaseModel):
    language: str = "it"
    topic: str = "free conversation"
    personality: str = "encouraging"
    started_at: str = ""

    @classmethod
    def create(
        cls,
        language: str = "it",
        topic: str = "free conversation",
        personality: str = "encouraging",
    ) -> SessionContext:
        return cls(
            language=language,
            topic=topic,
            personality=personality,
            started_at=datetime.now(timezone.utc).isoformat(),
        )


def parse_start_command(args: str) -> tuple[str | None, str | None, str | None]:
    """Parse '/start <topic> [as <personality>]'.

    Returns (topic, personality, error).
    If error is set, topic and personality are None.
    """
    args = args.strip()
    if not args:
        return None, None, None  # No args = show usage

    # Split on ' as ' (case-insensitive) to separate topic from personality
    # Use lower-cased version for splitting
    lower = args.lower()
    as_idx = lower.find(" as ")
    if as_idx >= 0:
        parts = [args[:as_idx], args[as_idx + 4:]]
    else:
        parts = [args]
    topic = parts[0].strip().lower()
    personality = parts[1].strip().lower() if len(parts) > 1 else "encouraging"

    # Validate topic
    if topic not in TOPICS:
        return None, None, f"Unknown topic: '{topic}'. Available: {', '.join(TOPICS)}"

    # Validate personality
    if personality not in PERSONALITIES:
        return None, None, (
            f"Unknown personality: '{personality}'. Available: {', '.join(PERSONALITIES.keys())}"
        )

    return topic, personality, None
