"""Comprehensive tests for config loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from language_learning.config import TutorConfig, load_config


# ---------------------------------------------------------------------------
# TutorConfig validation
# ---------------------------------------------------------------------------


class TestTutorConfig:
    def test_minimal_valid_config(self):
        cfg = TutorConfig(model="openai/gpt-4o")
        assert cfg.model == "openai/gpt-4o"
        assert cfg.api_base is None
        assert cfg.api_key is None
        assert cfg.timeout == 30
        assert cfg.max_tokens == 1024

    def test_full_config(self):
        cfg = TutorConfig(
            model="anthropic/claude-sonnet-4-6",
            api_base="http://localhost:8000/v1",
            api_key="sk-test",
            timeout=60,
            max_tokens=2048,
        )
        assert cfg.api_base == "http://localhost:8000/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.timeout == 60
        assert cfg.max_tokens == 2048

    def test_ds4_style_config(self):
        cfg = TutorConfig(
            model="openai/deepseek-v4",
            api_base="http://10.0.4.105:8000/v1",
            api_key="not-needed",
        )
        assert cfg.model == "openai/deepseek-v4"
        assert cfg.api_base == "http://10.0.4.105:8000/v1"

    def test_anthropic_style_config(self):
        # No api_base, no api_key (key comes from env var)
        cfg = TutorConfig(model="anthropic/claude-sonnet-4-6")
        assert cfg.api_base is None
        assert cfg.api_key is None

    def test_openai_style_config(self):
        cfg = TutorConfig(model="openai/gpt-4o")
        assert cfg.model == "openai/gpt-4o"

    def test_empty_model_raises(self):
        with pytest.raises(Exception, match="model"):
            TutorConfig(model="")

    def test_blank_model_raises(self):
        with pytest.raises(Exception, match="model"):
            TutorConfig(model="   ")

    def test_zero_timeout_raises(self):
        with pytest.raises(Exception, match="timeout"):
            TutorConfig(model="openai/gpt-4o", timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(Exception, match="timeout"):
            TutorConfig(model="openai/gpt-4o", timeout=-5)

    def test_zero_max_tokens_raises(self):
        with pytest.raises(Exception, match="max_tokens"):
            TutorConfig(model="openai/gpt-4o", max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(Exception, match="max_tokens"):
            TutorConfig(model="openai/gpt-4o", max_tokens=-1)

    def test_unknown_extra_fields_ignored(self):
        cfg = TutorConfig.model_validate({"model": "openai/gpt-4o", "unknown_field": "ignored"})
        assert cfg.model == "openai/gpt-4o"
        assert not hasattr(cfg, "unknown_field")


# ---------------------------------------------------------------------------
# load_config from explicit path
# ---------------------------------------------------------------------------


class TestLoadConfigExplicitPath:
    def _write(self, path: Path, data: dict) -> None:
        path.write_text(yaml.dump(data))

    def test_loads_minimal_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"model": "openai/gpt-4o"})
        cfg = load_config(cfg_file)
        assert cfg.model == "openai/gpt-4o"

    def test_loads_full_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {
            "model": "anthropic/claude-sonnet-4-6",
            "api_base": "http://localhost:8000/v1",
            "api_key": "sk-test",
            "timeout": 45,
            "max_tokens": 512,
        })
        cfg = load_config(cfg_file)
        assert cfg.api_base == "http://localhost:8000/v1"
        assert cfg.timeout == 45
        assert cfg.max_tokens == 512

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_missing_required_model_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"timeout": 30})
        with pytest.raises(Exception):
            load_config(cfg_file)

    def test_invalid_yaml_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model: [\nbroken yaml")
        with pytest.raises(Exception):
            load_config(cfg_file)

    def test_yaml_non_mapping_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_config(cfg_file)

    def test_string_path_accepted(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"model": "openai/gpt-4o"})
        cfg = load_config(str(cfg_file))
        assert cfg.model == "openai/gpt-4o"

    def test_null_values_for_optional_fields(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"model": "openai/gpt-4o", "api_base": None, "api_key": None})
        cfg = load_config(cfg_file)
        assert cfg.api_base is None
        assert cfg.api_key is None

    def test_extra_fields_ignored(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"model": "openai/gpt-4o", "future_param": "value"})
        cfg = load_config(cfg_file)
        assert cfg.model == "openai/gpt-4o"

    def test_invalid_timeout_in_file_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write(cfg_file, {"model": "openai/gpt-4o", "timeout": -1})
        with pytest.raises(Exception):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# load_config auto-discovery
# ---------------------------------------------------------------------------


class TestLoadConfigAutoDiscovery:
    def test_discovers_local_config_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"model": "openai/gpt-4o"}))
        cfg = load_config()
        assert cfg.model == "openai/gpt-4o"

    def test_discovers_home_config(self, tmp_path, monkeypatch):
        # Patch home directory to tmp_path and move to a different dir
        monkeypatch.chdir(tmp_path / "..")
        home_cfg_dir = tmp_path / ".config" / "language-tutor"
        home_cfg_dir.mkdir(parents=True)
        (home_cfg_dir / "config.yaml").write_text(yaml.dump({"model": "anthropic/claude-sonnet-4-6"}))

        # Patch Path.home() to return tmp_path
        monkeypatch.setattr(
            "language_learning.config._SEARCH_PATHS",
            [
                tmp_path / "no_such_config.yaml",
                home_cfg_dir / "config.yaml",
            ],
        )
        cfg = load_config()
        assert cfg.model == "anthropic/claude-sonnet-4-6"

    def test_local_config_takes_precedence_over_home(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "config.yaml"
        local.write_text(yaml.dump({"model": "openai/gpt-4o"}))
        home_cfg = tmp_path / "home_config.yaml"
        home_cfg.write_text(yaml.dump({"model": "anthropic/claude-sonnet-4-6"}))
        monkeypatch.setattr(
            "language_learning.config._SEARCH_PATHS",
            [local, home_cfg],
        )
        cfg = load_config()
        assert cfg.model == "openai/gpt-4o"

    def test_raises_file_not_found_when_none_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "language_learning.config._SEARCH_PATHS",
            [tmp_path / "a.yaml", tmp_path / "b.yaml"],
        )
        with pytest.raises(FileNotFoundError, match="No config.yaml found"):
            load_config()

    def test_error_message_lists_searched_paths(self, tmp_path, monkeypatch):
        p1 = tmp_path / "a.yaml"
        p2 = tmp_path / "b.yaml"
        monkeypatch.setattr("language_learning.config._SEARCH_PATHS", [p1, p2])
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config()
        assert str(p1) in str(exc_info.value)
        assert str(p2) in str(exc_info.value)
