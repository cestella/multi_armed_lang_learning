"""Discord bot frontend for the adaptive language tutor."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import discord
import yaml
from discord import app_commands
from discord.ext import commands
from pydantic import BaseModel, field_validator

from language_learning.config import TutorConfig, load_arms, load_config, load_story_topics, resolve_config_dir
from language_learning.controller.registry import ControllerRegistry
from language_learning.models.arms import Arm
from language_learning.models.state import FeedbackPanel

logger = logging.getLogger(__name__)

# Emoji constants for star ratings
STAR_EMOJIS = ("1\u20e3", "2\u20e3", "3\u20e3", "4\u20e3", "5\u20e3")
STAR_EMOJI_TO_INT: dict[str, int] = {e: i + 1 for i, e in enumerate(STAR_EMOJIS)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DiscordConfig(BaseModel):
    """Discord bot configuration loaded from discord.yaml."""

    bot_token: str | None = None
    guild_id: int | None = None
    allowed_channel_ids: list[int] = []
    default_language: str = "it"
    data_dir: str = "./lang_data"

    @field_validator("default_language")
    @classmethod
    def language_supported(cls, v: str) -> str:
        if v not in ("it", "es"):
            raise ValueError(f"Unsupported language: {v!r} (must be 'it' or 'es')")
        return v

    model_config = {"extra": "ignore"}


def load_discord_config(path: Path) -> DiscordConfig:
    """Load DiscordConfig from a YAML file, falling back to env vars."""
    if path.exists():
        with open(path) as f:
            raw: Any = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raw = {}
    else:
        raw = {}

    config = DiscordConfig.model_validate(raw)

    # Fall back to env var for bot token
    if not config.bot_token:
        config.bot_token = os.environ.get("DISCORD_BOT_TOKEN")

    return config


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class LanguageTutorBot(commands.Bot):
    """Discord bot that routes messages through the ControllerRegistry."""

    def __init__(
        self,
        discord_config: DiscordConfig,
        tutor_config: TutorConfig,
        arms: list[Arm],
        story_topics: list[str] | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True

        super().__init__(command_prefix="!", intents=intents)

        self.discord_config = discord_config
        self.tutor_config = tutor_config
        self.arms = arms
        self.story_topics: list[str] = story_topics or []
        self.registry: ControllerRegistry | None = None

        # discord user ID -> language
        self._active_sessions: dict[int, str] = {}
        # discord message ID -> (discord_user_id, turn_id)
        self._rating_map: dict[int, tuple[int, str]] = {}

    async def setup_hook(self) -> None:
        self.registry = ControllerRegistry(
            config=self.tutor_config,
            data_dir=self.discord_config.data_dir,
            arms=self.arms,
            default_language=self.discord_config.default_language,
        )

        # Register slash commands
        self.tree.add_command(_start_cmd)
        self.tree.add_command(_story_cmd)
        self.tree.add_command(_stop_cmd)
        self.tree.add_command(_stats_cmd)
        self.tree.add_command(_cefr_cmd)
        self.tree.add_command(_why_cmd)
        self.tree.add_command(_language_cmd)

        if self.discord_config.guild_id:
            guild = discord.Object(id=self.discord_config.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    def _channel_allowed(self, channel_id: int) -> bool:
        if not self.discord_config.allowed_channel_ids:
            return True
        return channel_id in self.discord_config.allowed_channel_ids

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.author.id not in self._active_sessions:
            return
        if not self._channel_allowed(message.channel.id):
            return

        language = self._active_sessions[message.author.id]
        controller = await self.registry.get(str(message.author.id), language)

        async with message.channel.typing():
            await controller.process_turn(message.content)

        await self._send_turn_response(message, controller, message.author.id)

    async def _send_turn_response(
        self,
        channel_or_message: discord.Message | discord.abc.Messageable,
        controller: Any,
        user_id: int,
    ) -> None:
        # Determine channel and whether we have a user message to reply to
        if isinstance(channel_or_message, discord.Message):
            user_message = channel_or_message
            channel = user_message.channel
        else:
            user_message = None
            channel = channel_or_message

        # Send coach feedback as a reply to the user's message
        embed = _build_feedback_embed(controller.app_state.feedback_panel)
        if embed is not None and user_message is not None:
            await user_message.reply(embed=embed, mention_author=False)

        # Extract last assistant message
        reply_text = ""
        for msg in reversed(controller.app_state.chat_messages):
            if msg.role == "assistant":
                reply_text = msg.text
                break

        # Send the tutor's response as a separate message
        sent = await channel.send(content=reply_text)

        # Add star reactions if there's a pending rating
        if controller.app_state.pending_rating_turn_id:
            turn_id = controller.app_state.pending_rating_turn_id
            for emoji in STAR_EMOJIS:
                await sent.add_reaction(emoji)
            self._rating_map[sent.id] = (user_id, turn_id)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        # Skip bot's own reactions
        if payload.user_id == self.user.id:
            return

        if payload.message_id not in self._rating_map:
            return

        emoji_str = str(payload.emoji)
        if emoji_str not in STAR_EMOJI_TO_INT:
            return

        user_id, turn_id = self._rating_map[payload.message_id]

        # Only the user who owns the session can rate
        if payload.user_id != user_id:
            return

        stars = STAR_EMOJI_TO_INT[emoji_str]
        language = self._active_sessions.get(user_id)
        if not language:
            return

        controller = await self.registry.get(str(user_id), language)
        controller.submit_engagement_rating(turn_id, stars)

        # Remove all star reactions from the message
        channel = self.get_channel(payload.channel_id)
        if channel:
            try:
                message = await channel.fetch_message(payload.message_id)
                for emoji in STAR_EMOJIS:
                    await message.clear_reaction(emoji)
            except discord.HTTPException:
                logger.warning("Failed to clear reactions", exc_info=True)

        del self._rating_map[payload.message_id]


# ---------------------------------------------------------------------------
# Slash commands (module-level, attached in setup_hook)
# ---------------------------------------------------------------------------


@app_commands.command(name="start", description="Start a language tutoring session")
@app_commands.describe(
    topic="Conversation topic (default: free conversation)",
    personality="Tutor personality (default: encouraging)",
    cefr_override="Override CEFR level (e.g. A1, A2, B1)",
)
async def _start_cmd(
    interaction: discord.Interaction,
    topic: str = "free conversation",
    personality: str = "encouraging",
    cefr_override: str | None = None,
) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]

    if not bot._channel_allowed(interaction.channel_id):
        await interaction.response.send_message(
            "This bot is not enabled in this channel.", ephemeral=True
        )
        return

    language = bot._active_sessions.get(
        interaction.user.id, bot.discord_config.default_language
    )
    bot._active_sessions[interaction.user.id] = language

    await interaction.response.defer()

    controller = await bot.registry.get(str(interaction.user.id), language)
    await controller.start_session(topic, personality, cefr_override=cefr_override)

    await bot._send_turn_response(interaction.channel, controller, interaction.user.id)
    await interaction.followup.send(
        f"Session started! Language: **{language}**, Topic: **{topic}**",
        ephemeral=True,
    )


@app_commands.command(name="story", description="Start a session with a short story and question")
@app_commands.describe(
    topic="Story topic (default: random from story_topics.yaml)",
    personality="Tutor personality (default: encouraging)",
    cefr_override="Override CEFR level (e.g. A1, A2, B1)",
)
async def _story_cmd(
    interaction: discord.Interaction,
    topic: str | None = None,
    personality: str = "encouraging",
    cefr_override: str | None = None,
) -> None:
    import random

    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]

    if not bot._channel_allowed(interaction.channel_id):
        await interaction.response.send_message(
            "This bot is not enabled in this channel.", ephemeral=True
        )
        return

    language = bot._active_sessions.get(
        interaction.user.id, bot.discord_config.default_language
    )
    bot._active_sessions[interaction.user.id] = language

    # Pick a random topic if none provided
    if topic is None:
        if bot.story_topics:
            topic = random.choice(bot.story_topics)
        else:
            topic = "daily life"

    await interaction.response.defer()

    controller = await bot.registry.get(str(interaction.user.id), language)
    await controller.start_story_session(topic, personality, cefr_override=cefr_override)

    # Send story text directly (no feedback embed or star reactions for the story itself)
    reply_text = ""
    for msg in reversed(controller.app_state.chat_messages):
        if msg.role == "assistant":
            reply_text = msg.text
            break

    await interaction.channel.send(content=reply_text)
    await interaction.followup.send(
        f"Story session started! Language: **{language}**, Topic: **{topic}**",
        ephemeral=True,
    )


@app_commands.command(name="stop", description="End the current tutoring session")
async def _stop_cmd(interaction: discord.Interaction) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]
    user_id = interaction.user.id

    if user_id not in bot._active_sessions:
        await interaction.response.send_message("No active session.", ephemeral=True)
        return

    language = bot._active_sessions.pop(user_id)
    await bot.registry.remove(str(user_id), language)
    await interaction.response.send_message("Session ended.", ephemeral=True)


@app_commands.command(name="stats", description="Show bandit statistics")
async def _stats_cmd(interaction: discord.Interaction) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]
    user_id = interaction.user.id

    language = bot._active_sessions.get(user_id, bot.discord_config.default_language)
    controller = await bot.registry.get(str(user_id), language)
    text = controller.get_stats()
    await interaction.response.send_message(f"```\n{text}\n```", ephemeral=True)


@app_commands.command(name="cefr", description="Show CEFR level estimate")
async def _cefr_cmd(interaction: discord.Interaction) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]
    user_id = interaction.user.id

    language = bot._active_sessions.get(user_id, bot.discord_config.default_language)
    controller = await bot.registry.get(str(user_id), language)
    text = controller.get_cefr_summary()
    await interaction.response.send_message(f"```\n{text}\n```", ephemeral=True)


@app_commands.command(name="why", description="Explain arm selection reasoning")
async def _why_cmd(interaction: discord.Interaction) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]
    user_id = interaction.user.id

    language = bot._active_sessions.get(user_id, bot.discord_config.default_language)
    controller = await bot.registry.get(str(user_id), language)
    text = controller.get_why()
    await interaction.response.send_message(f"```\n{text}\n```", ephemeral=True)


@app_commands.command(name="language", description="Switch target language")
@app_commands.describe(lang="Target language code")
@app_commands.choices(
    lang=[
        app_commands.Choice(name="Italian", value="it"),
        app_commands.Choice(name="Spanish", value="es"),
    ]
)
async def _language_cmd(
    interaction: discord.Interaction,
    lang: app_commands.Choice[str],
) -> None:
    bot: LanguageTutorBot = interaction.client  # type: ignore[assignment]
    user_id = interaction.user.id

    if user_id not in bot._active_sessions:
        await interaction.response.send_message("No active session.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    old_language = bot._active_sessions[user_id]
    new_language = lang.value
    controller = await bot.registry.get(str(user_id), old_language)
    await controller.switch_language(new_language)
    bot._active_sessions[user_id] = new_language

    await interaction.followup.send(
        f"Switched to **{lang.name}** ({new_language}).", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_feedback_embed(feedback: FeedbackPanel) -> discord.Embed | None:
    """Build a Discord embed from the feedback panel."""
    parts: list[str] = []
    if feedback.praise:
        parts.append(f"\u2713 {feedback.praise}")
    if feedback.fix_one:
        parts.append(f"\u270e {feedback.fix_one}")
    if feedback.micro_rule:
        parts.append(f"\U0001f4cc {feedback.micro_rule}")
    if feedback.next_nudge:
        parts.append(f"\u2192 {feedback.next_nudge}")
    if feedback.hint_phrase:
        parts.append(f"\U0001f4a1 {feedback.hint_phrase}")
    if feedback.retry_prompt:
        parts.append(f"\u21ba {feedback.retry_prompt}")

    if not parts:
        return None

    embed = discord.Embed(
        title="Coach",
        description="\n".join(parts),
        color=0x0B93F6,
    )
    return embed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_bot(config_dir: Path, data_dir: str | None = None) -> None:
    """Load configs, create bot, and run."""
    resolved_config_dir = resolve_config_dir(str(config_dir))
    tutor_config = load_config(config_dir=resolved_config_dir)
    arms_data = load_arms(resolved_config_dir)
    arms = [Arm(**a) for a in arms_data]

    discord_config = load_discord_config(config_dir / "discord.yaml")

    if data_dir is not None:
        discord_config.data_dir = data_dir

    token = discord_config.bot_token
    if not token:
        raise RuntimeError(
            "No bot token found. Set bot_token in discord.yaml or "
            "the DISCORD_BOT_TOKEN environment variable."
        )

    story_topics = load_story_topics(resolved_config_dir)
    bot = LanguageTutorBot(discord_config, tutor_config, arms, story_topics=story_topics)
    bot.run(token, log_handler=logging.StreamHandler())
