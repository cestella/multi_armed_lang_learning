"""Progress report generation — static charts and markdown from persisted files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from language_learning.core.storage import read_json, read_jsonl

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Stable numeric mapping for CEFR labels
CEFR_NUMERIC: dict[str, float] = {
    "A1": 1.0,
    "A1+": 1.5,
    "A2": 2.0,
    "A2+": 2.5,
    "A2+/B1-": 2.75,
    "B1-": 2.9,
    "B1": 3.0,
    "B1+": 3.5,
    "B1+/B2-": 3.75,
    "B2": 4.0,
}

CEFR_DOMAINS = ["conversation", "description", "narration", "opinion", "comprehension"]


def generate_progress_reports(data_dir: str | Path, language: str) -> Path:
    """Generate all progress reports for a language. Returns the report directory."""
    data_dir = Path(data_dir)
    report_dir = data_dir / "reports" / language
    report_dir.mkdir(parents=True, exist_ok=True)

    cefr_history = _load_cefr_history(data_dir, language)
    skill_history = _load_skill_history(data_dir, language)
    session_metrics = _load_session_metrics(data_dir, language)
    cefr_state = read_json(data_dir / "state" / language / "cefr_state.json")
    skill_state = read_json(data_dir / "state" / language / "skill_state.json")

    if cefr_history:
        build_cefr_chart(cefr_history, report_dir / "cefr_progress.png")

    if skill_history:
        build_skill_mastery_chart(skill_history, report_dir / "skill_mastery.png")
        build_scaffold_chart(skill_history, report_dir / "scaffold_trend.png")

    if session_metrics:
        build_attempt_avoidance_chart(
            session_metrics, report_dir / "attempt_avoidance.png"
        )

    write_progress_summary(
        report_dir / "progress_summary.md",
        language,
        cefr_state,
        skill_state,
        cefr_history,
        skill_history,
        session_metrics,
    )

    return report_dir


# --- Data loading ---


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp, tolerating various formats."""
    # Strip trailing Z and handle timezone
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _load_cefr_history(data_dir: Path, language: str) -> list[dict]:
    return read_jsonl(data_dir / "state" / language / "cefr_history.jsonl")


def _load_skill_history(data_dir: Path, language: str) -> list[dict]:
    return read_jsonl(data_dir / "state" / language / "skill_history.jsonl")


def _load_session_metrics(data_dir: Path, language: str) -> list[dict]:
    return read_jsonl(data_dir / "state" / language / "session_metrics.jsonl")


# --- Chart builders ---


def build_cefr_chart(cefr_history: list[dict], output_path: Path) -> None:
    """CEFR level over time by domain."""
    if not cefr_history:
        return

    timestamps = [_parse_ts(r["ts"]) for r in cefr_history]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Overall
    overall_vals = [CEFR_NUMERIC.get(r.get("overall", "A1"), 1.0) for r in cefr_history]
    ax.plot(timestamps, overall_vals, linewidth=2.5, label="overall", color="black")

    # Per domain
    for domain in CEFR_DOMAINS:
        vals = []
        for r in cefr_history:
            domains = r.get("domains", {})
            label = domains.get(domain)
            if label:
                vals.append(CEFR_NUMERIC.get(label, 1.0))
            else:
                vals.append(None)
        # Only plot if there's data
        if any(v is not None for v in vals):
            ax.plot(timestamps, vals, linewidth=1.2, label=domain, alpha=0.7)

    ax.set_ylabel("CEFR Level")
    ax.set_xlabel("Time")
    ax.set_title("CEFR Progress Over Time")

    # Set y-axis to show CEFR labels
    label_positions = sorted(set(CEFR_NUMERIC.values()))
    label_names = {v: k for k, v in CEFR_NUMERIC.items()}
    ax.set_yticks(label_positions)
    ax.set_yticklabels([label_names.get(p, "") for p in label_positions])
    ax.set_ylim(0.8, 4.2)

    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_skill_mastery_chart(skill_history: list[dict], output_path: Path) -> None:
    """Skill mastery over time — top 5 most active/relevant skills."""
    if not skill_history:
        return

    timestamps = [_parse_ts(r["ts"]) for r in skill_history]
    skills = _select_top_skills(skill_history, "mastery", 5)

    if not skills:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for skill_name in skills:
        vals = [
            r.get("skills", {}).get(skill_name, {}).get("mastery")
            for r in skill_history
        ]
        ax.plot(timestamps, vals, linewidth=1.5, label=skill_name)

    ax.set_ylabel("Mastery (0-1)")
    ax.set_xlabel("Time")
    ax.set_title("Skill Mastery Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_scaffold_chart(skill_history: list[dict], output_path: Path) -> None:
    """Scaffold need over time — top 5 skills."""
    if not skill_history:
        return

    timestamps = [_parse_ts(r["ts"]) for r in skill_history]
    skills = _select_top_skills(skill_history, "scaffold_need", 5)

    if not skills:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for skill_name in skills:
        vals = [
            r.get("skills", {}).get(skill_name, {}).get("scaffold_need")
            for r in skill_history
        ]
        ax.plot(timestamps, vals, linewidth=1.5, label=skill_name)

    ax.set_ylabel("Scaffold Need (0-1)")
    ax.set_xlabel("Time")
    ax.set_title("Scaffold Need Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_attempt_avoidance_chart(
    session_metrics: list[dict], output_path: Path
) -> None:
    """Attempt vs avoidance trend — rolling average."""
    if not session_metrics:
        return

    timestamps = [_parse_ts(r["ts"]) for r in session_metrics]
    attempt_vals = [1.0 if r.get("target_attempted") else 0.0 for r in session_metrics]
    avoidance_vals = [
        1.0 if r.get("avoidance") == "strong" else 0.0 for r in session_metrics
    ]
    success_map = {"no": 0.0, "partial": 0.5, "yes": 1.0}
    success_vals = [
        success_map.get(r.get("target_success", "no"), 0.0) for r in session_metrics
    ]

    window = min(10, len(session_metrics))
    attempt_smooth = _rolling_avg(attempt_vals, window)
    avoidance_smooth = _rolling_avg(avoidance_vals, window)
    success_smooth = _rolling_avg(success_vals, window)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timestamps, attempt_smooth, linewidth=1.5, label="attempt rate")
    ax.plot(timestamps, avoidance_smooth, linewidth=1.5, label="strong avoidance rate")
    ax.plot(
        timestamps,
        success_smooth,
        linewidth=1.5,
        label="success rate",
        linestyle="--",
    )

    ax.set_ylabel("Rate (0-1)")
    ax.set_xlabel("Time")
    ax.set_title("Attempt vs Avoidance Trend")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --- Markdown summary ---


def write_progress_summary(
    output_path: Path,
    language: str,
    cefr_state: dict | None,
    skill_state: dict | None,
    cefr_history: list[dict],
    skill_history: list[dict],
    session_metrics: list[dict],
) -> None:
    """Generate a compact markdown progress summary."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lang_upper = language.upper()

    lines = [
        f"# Progress Report ({lang_upper})",
        "",
        f"Generated: {now}",
        "",
    ]

    # CEFR summary
    if cefr_state:
        overall = cefr_state.get("overall_estimate", "A1")
        confidence = cefr_state.get("confidence", {})
        domains = cefr_state.get("domains", {})
        overall_conf = confidence.get("overall", 0.0)

        lines.append(f"## CEFR Estimate: {overall} (confidence: {overall_conf:.2f})")
        lines.append("")

        # Find strongest and weakest domains
        domain_scores = {}
        for domain in CEFR_DOMAINS:
            if domain in domains:
                domain_scores[domain] = CEFR_NUMERIC.get(domains[domain], 1.0)

        if domain_scores:
            strongest = max(domain_scores, key=domain_scores.get)
            weakest = min(domain_scores, key=domain_scores.get)
            lines.append(
                f"- **Strongest domain**: {strongest} ({domains.get(strongest, 'A1')})"
            )
            lines.append(
                f"- **Weakest domain**: {weakest} ({domains.get(weakest, 'A1')})"
            )
            lines.append("")

    # Skill analysis
    if skill_state:
        skills = skill_state.get("skills", {})
        if skills:
            # Sort by mastery
            sorted_skills = sorted(
                skills.items(), key=lambda x: x[1].get("mastery", 0.5)
            )

            lines.append("## Skills")
            lines.append("")

            # Top improving (highest mastery)
            top = sorted_skills[-3:]
            lines.append("**Strongest skills:**")
            for name, stats in reversed(top):
                m = stats.get("mastery", 0.5)
                lines.append(f"- {name}: mastery {m:.2f}")

            lines.append("")

            # Most needing work (lowest mastery)
            bottom = sorted_skills[:3]
            lines.append("**Needs work:**")
            for name, stats in bottom:
                m = stats.get("mastery", 0.5)
                sn = stats.get("scaffold_need", 0.3)
                lines.append(f"- {name}: mastery {m:.2f}, scaffold need {sn:.2f}")

            lines.append("")

    # Session stats
    if session_metrics:
        total_turns = len(session_metrics)
        attempted = sum(1 for r in session_metrics if r.get("target_attempted"))
        avoided = sum(1 for r in session_metrics if r.get("avoidance") == "strong")
        lines.append("## Session Statistics")
        lines.append("")
        lines.append(f"- Total evaluated turns: {total_turns}")
        lines.append(f"- Target attempted: {attempted} ({attempted/total_turns:.0%})")
        lines.append(
            f"- Strong avoidance: {avoided} ({avoided/total_turns:.0%})"
        )
        lines.append("")

    # Chart references
    lines.append("## Charts")
    lines.append("")
    if cefr_history:
        lines.append("![CEFR Progress](cefr_progress.png)")
        lines.append("")
    if skill_history:
        lines.append("![Skill Mastery](skill_mastery.png)")
        lines.append("")
        lines.append("![Scaffold Trend](scaffold_trend.png)")
        lines.append("")
    if session_metrics:
        lines.append("![Attempt vs Avoidance](attempt_avoidance.png)")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# --- Helpers ---


def _select_top_skills(
    skill_history: list[dict], metric: str, n: int
) -> list[str]:
    """Select top N skills by variance in a metric (most changing = most interesting)."""
    if not skill_history:
        return []

    # Collect all skill names
    all_skills: set[str] = set()
    for record in skill_history:
        all_skills.update(record.get("skills", {}).keys())

    # Compute variance for each skill
    variances: dict[str, float] = {}
    for skill_name in all_skills:
        vals = []
        for record in skill_history:
            v = record.get("skills", {}).get(skill_name, {}).get(metric)
            if v is not None:
                vals.append(v)
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            variances[skill_name] = variance
        elif vals:
            variances[skill_name] = 0.0

    # Sort by variance (most changing first), break ties by name
    sorted_skills = sorted(variances.keys(), key=lambda s: (-variances[s], s))
    return sorted_skills[:n]


def _rolling_avg(values: list[float], window: int) -> list[float]:
    """Compute a simple rolling average."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _format_date_axis(ax: plt.Axes) -> None:
    """Format x-axis for date display."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig = ax.get_figure()
    fig.autofmt_xdate(rotation=30, ha="right")
