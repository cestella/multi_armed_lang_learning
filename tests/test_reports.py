"""Tests for progress report generation and history persistence."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from language_learning.controller.controller import Controller
from language_learning.core.storage import append_jsonl, read_jsonl, write_json
from language_learning.reporting.progress_reports import (
    CEFR_NUMERIC,
    build_attempt_avoidance_chart,
    build_cefr_chart,
    build_scaffold_chart,
    build_skill_mastery_chart,
    generate_progress_reports,
    write_progress_summary,
)


@pytest.fixture
def data_dir(tmp_path):
    """Create a data directory with arms config."""
    arms_src = Path(__file__).parent.parent / "arms" / "arms.yaml"
    arms_dst = tmp_path / "arms" / "arms.yaml"
    arms_dst.parent.mkdir(parents=True)
    shutil.copy(arms_src, arms_dst)
    return tmp_path


def _ts(offset_minutes: int = 0) -> str:
    """Generate a UTC ISO timestamp."""
    from datetime import timedelta

    dt = datetime(2026, 3, 6, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=offset_minutes
    )
    return dt.isoformat()


def _seed_cefr_history(data_dir: Path, language: str, n: int = 10) -> None:
    path = data_dir / "state" / language / "cefr_history.jsonl"
    for i in range(n):
        record = {
            "ts": _ts(i * 5),
            "overall": "A2" if i < 5 else "A2+",
            "domains": {
                "conversation": "A2" if i < 5 else "B1-",
                "description": "A2",
                "narration": "A1+" if i < 3 else "A2",
                "opinion": "A1",
                "comprehension": "A1+",
            },
            "confidence": {
                "overall": 0.2 + i * 0.05,
                "conversation": 0.3 + i * 0.04,
            },
        }
        append_jsonl(path, record)


def _seed_skill_history(data_dir: Path, language: str, n: int = 10) -> None:
    path = data_dir / "state" / language / "skill_history.jsonl"
    for i in range(n):
        record = {
            "ts": _ts(i * 5),
            "skills": {
                "past_narration": {
                    "mastery": 0.3 + i * 0.04,
                    "scaffold_need": 0.5 - i * 0.03,
                },
                "vocabulary_description": {
                    "mastery": 0.5 + i * 0.02,
                    "scaffold_need": 0.3 - i * 0.01,
                },
                "agreement": {
                    "mastery": 0.4 + i * 0.03,
                    "scaffold_need": 0.4 - i * 0.02,
                },
            },
        }
        append_jsonl(path, record)


def _seed_session_metrics(data_dir: Path, language: str, n: int = 10) -> None:
    path = data_dir / "state" / language / "session_metrics.jsonl"
    for i in range(n):
        record = {
            "ts": _ts(i * 5),
            "target_attempted": i % 3 != 0,
            "target_success": ["no", "partial", "yes"][i % 3],
            "avoidance": "none" if i % 5 != 0 else "strong",
            "fluency_proxy": 0.5 + i * 0.03,
            "novelty_proxy": 0.4 + i * 0.02,
            "arm_id": "narrative_yesterday",
        }
        append_jsonl(path, record)


def _seed_all(data_dir: Path, language: str = "it", n: int = 10) -> None:
    _seed_cefr_history(data_dir, language, n)
    _seed_skill_history(data_dir, language, n)
    _seed_session_metrics(data_dir, language, n)

    # Also seed current state snapshots
    state_dir = data_dir / "state" / language
    state_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        state_dir / "cefr_state.json",
        {
            "schema_version": 1,
            "language": language,
            "overall_estimate": "A2+",
            "domains": {
                "conversation": "B1-",
                "description": "A2",
                "narration": "A2",
                "opinion": "A1",
                "comprehension": "A1+",
            },
            "confidence": {"overall": 0.45, "conversation": 0.6},
            "evidence": {},
            "last_updated": _ts(50),
        },
    )

    write_json(
        state_dir / "skill_state.json",
        {
            "schema_version": 1,
            "language": language,
            "skills": {
                "past_narration": {"mastery": 0.7, "scaffold_need": 0.2, "confidence": 0.5},
                "vocabulary_description": {"mastery": 0.6, "scaffold_need": 0.2, "confidence": 0.4},
                "agreement": {"mastery": 0.55, "scaffold_need": 0.25, "confidence": 0.3},
            },
            "updated_at": _ts(50),
        },
    )


class TestGenerateProgressReports:
    def test_generates_all_files(self, tmp_path):
        """Full generation produces all expected output files."""
        _seed_all(tmp_path)
        report_dir = generate_progress_reports(tmp_path, "it")

        assert (report_dir / "progress_summary.md").exists()
        assert (report_dir / "cefr_progress.png").exists()
        assert (report_dir / "skill_mastery.png").exists()
        assert (report_dir / "scaffold_trend.png").exists()
        assert (report_dir / "attempt_avoidance.png").exists()

    def test_generates_with_no_data(self, tmp_path):
        """No history files — still produces summary without crashing."""
        report_dir = generate_progress_reports(tmp_path, "it")
        assert (report_dir / "progress_summary.md").exists()
        # Charts should not exist since there's no data
        assert not (report_dir / "cefr_progress.png").exists()

    def test_report_dir_is_correct(self, tmp_path):
        """Reports written to reports/<lang>/."""
        _seed_all(tmp_path)
        report_dir = generate_progress_reports(tmp_path, "it")
        assert report_dir == tmp_path / "reports" / "it"

    def test_idempotent(self, tmp_path):
        """Running twice overwrites cleanly."""
        _seed_all(tmp_path)
        generate_progress_reports(tmp_path, "it")
        generate_progress_reports(tmp_path, "it")
        assert (tmp_path / "reports" / "it" / "progress_summary.md").exists()


class TestCefrChart:
    def test_builds_png(self, tmp_path):
        output = tmp_path / "cefr.png"
        history = [
            {"ts": _ts(0), "overall": "A1", "domains": {"conversation": "A1"}},
            {"ts": _ts(5), "overall": "A2", "domains": {"conversation": "A2"}},
            {"ts": _ts(10), "overall": "A2+", "domains": {"conversation": "B1-"}},
        ]
        build_cefr_chart(history, output)
        assert output.exists()
        assert output.stat().st_size > 1000  # non-trivial PNG

    def test_empty_history_no_crash(self, tmp_path):
        output = tmp_path / "cefr.png"
        build_cefr_chart([], output)
        assert not output.exists()


class TestSkillMasteryChart:
    def test_builds_png(self, tmp_path):
        output = tmp_path / "mastery.png"
        history = [
            {"ts": _ts(i), "skills": {"past_narration": {"mastery": 0.3 + i * 0.05}}}
            for i in range(5)
        ]
        build_skill_mastery_chart(history, output)
        assert output.exists()

    def test_empty_history_no_crash(self, tmp_path):
        output = tmp_path / "mastery.png"
        build_skill_mastery_chart([], output)
        assert not output.exists()


class TestScaffoldChart:
    def test_builds_png(self, tmp_path):
        output = tmp_path / "scaffold.png"
        history = [
            {
                "ts": _ts(i),
                "skills": {"past_narration": {"scaffold_need": 0.5 - i * 0.05}},
            }
            for i in range(5)
        ]
        build_scaffold_chart(history, output)
        assert output.exists()


class TestAttemptAvoidanceChart:
    def test_builds_png(self, tmp_path):
        output = tmp_path / "attempt.png"
        metrics = [
            {
                "ts": _ts(i),
                "target_attempted": True,
                "target_success": "yes",
                "avoidance": "none",
            }
            for i in range(5)
        ]
        build_attempt_avoidance_chart(metrics, output)
        assert output.exists()


class TestProgressSummary:
    def test_contains_key_sections(self, tmp_path):
        output = tmp_path / "summary.md"
        _seed_all(tmp_path)

        cefr_state = json.loads(
            (tmp_path / "state" / "it" / "cefr_state.json").read_text()
        )
        skill_state = json.loads(
            (tmp_path / "state" / "it" / "skill_state.json").read_text()
        )
        cefr_history = read_jsonl(tmp_path / "state" / "it" / "cefr_history.jsonl")
        skill_history = read_jsonl(tmp_path / "state" / "it" / "skill_history.jsonl")
        session_metrics = read_jsonl(
            tmp_path / "state" / "it" / "session_metrics.jsonl"
        )

        write_progress_summary(
            output,
            "it",
            cefr_state,
            skill_state,
            cefr_history,
            skill_history,
            session_metrics,
        )

        content = output.read_text()
        assert "Progress Report (IT)" in content
        assert "CEFR Estimate" in content
        assert "Strongest domain" in content
        assert "Weakest domain" in content
        assert "Strongest skills" in content
        assert "Needs work" in content
        assert "Session Statistics" in content
        assert "cefr_progress.png" in content
        assert "skill_mastery.png" in content

    def test_summary_with_no_data(self, tmp_path):
        output = tmp_path / "summary.md"
        write_progress_summary(output, "it", None, None, [], [], [])
        content = output.read_text()
        assert "Progress Report (IT)" in content


class TestCefrNumericMapping:
    def test_all_labels_mapped(self):
        expected = {
            "A1", "A1+", "A2", "A2+", "A2+/B1-",
            "B1-", "B1", "B1+", "B1+/B2-", "B2",
        }
        assert set(CEFR_NUMERIC.keys()) == expected

    def test_monotonic(self):
        """Numeric values increase with CEFR level."""
        labels = ["A1", "A1+", "A2", "A2+", "A2+/B1-", "B1-", "B1", "B1+", "B1+/B2-", "B2"]
        values = [CEFR_NUMERIC[l] for l in labels]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], f"{labels[i]} >= {labels[i+1]}"


class TestHistoryPersistence:
    """Test that the controller appends history records during evaluation."""

    @pytest.fixture
    def controller(self, data_dir):
        from unittest.mock import AsyncMock, MagicMock
        from language_learning.config import TutorConfig
        from language_learning.controller.llm_client import LLMClient
        from language_learning.models.evaluation import EvaluationResult

        cfg = TutorConfig(model="openai/gpt-4o", api_key="test")
        ctrl = Controller(language="it", config=cfg, data_dir=str(data_dir))
        ctrl.llm_client = MagicMock(spec=LLMClient)
        ctrl.llm_client.evaluate = AsyncMock(return_value=EvaluationResult(
            target_attempted=True,
            target_success="partial",
            avoidance="none",
            fluency_proxy=0.7,
            novelty_proxy=0.6,
        ))
        ctrl.llm_client.generate_response = AsyncMock(return_value="Ottimo!")
        ctrl.llm_client.initiate = AsyncMock(return_value="Ciao!")
        return ctrl

    async def test_evaluation_appends_skill_history(self, controller, data_dir):
        """skill_history.jsonl written after evaluation."""
        await controller.initialize()
        await controller.process_turn("Sono andato al cinema")

        history_path = data_dir / "state" / "it" / "skill_history.jsonl"
        assert history_path.exists()
        records = read_jsonl(history_path)
        assert len(records) >= 1
        assert "skills" in records[0]
        assert "ts" in records[0]

    async def test_evaluation_appends_cefr_history(self, controller, data_dir):
        """cefr_history.jsonl written after evaluation."""
        await controller.initialize()
        await controller.process_turn("Ho mangiato la pasta")

        history_path = data_dir / "state" / "it" / "cefr_history.jsonl"
        assert history_path.exists()
        records = read_jsonl(history_path)
        assert len(records) >= 1
        assert "overall" in records[0]
        assert "domains" in records[0]

    async def test_evaluation_appends_session_metrics(self, controller, data_dir):
        """session_metrics.jsonl written after evaluation."""
        controller.llm_client.evaluate.return_value.__class__ = type(
            controller.llm_client.evaluate.return_value
        )
        from unittest.mock import AsyncMock
        from language_learning.models.evaluation import EvaluationResult
        controller.llm_client.evaluate = AsyncMock(return_value=EvaluationResult(
            target_attempted=True,
            target_success="partial",
            avoidance="weak",
            fluency_proxy=0.6,
            novelty_proxy=0.4,
        ))

        await controller.initialize()
        await controller.process_turn("Ieri ho lavorato")

        history_path = data_dir / "state" / "it" / "session_metrics.jsonl"
        assert history_path.exists()
        records = read_jsonl(history_path)
        assert len(records) >= 1
        assert records[0]["target_attempted"] is True
        assert records[0]["avoidance"] == "weak"
        assert records[0]["arm_id"] is not None


class TestGenerateReportsFromController:
    """Test generate_reports integration on controller."""

    @pytest.fixture
    def controller(self, data_dir):
        from unittest.mock import AsyncMock, MagicMock
        from language_learning.config import TutorConfig
        from language_learning.controller.llm_client import LLMClient
        from language_learning.models.evaluation import EvaluationResult

        cfg = TutorConfig(model="openai/gpt-4o", api_key="test")
        ctrl = Controller(language="it", config=cfg, data_dir=str(data_dir))
        ctrl.llm_client = MagicMock(spec=LLMClient)
        ctrl.llm_client.evaluate = AsyncMock(return_value=EvaluationResult(
            target_attempted=True, target_success="partial",
            fluency_proxy=0.6, novelty_proxy=0.5,
        ))
        ctrl.llm_client.generate_response = AsyncMock(return_value="Ottimo!")
        ctrl.llm_client.initiate = AsyncMock(return_value="Ciao!")
        return ctrl

    async def test_generate_reports_best_effort(self, controller, data_dir):
        """generate_reports does not raise even with no data."""
        await controller.initialize()
        controller.generate_reports()
        assert (data_dir / "reports" / "it" / "progress_summary.md").exists()

    async def test_generate_reports_after_evaluation(self, controller, data_dir):
        """Reports generated after evaluation contain charts."""
        await controller.initialize()

        for text in ["Ciao", "Come stai", "Bene grazie"]:
            await controller.process_turn(text)

        controller.generate_reports()

        report_dir = data_dir / "reports" / "it"
        assert (report_dir / "progress_summary.md").exists()
        assert (report_dir / "cefr_progress.png").exists()
        assert (report_dir / "attempt_avoidance.png").exists()
