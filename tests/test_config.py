"""Tests for configuration file support."""

import json
import os
import tempfile

import pytest

from pathfinder_sdk.config import load_config


class TestLoadConfig:
    def test_empty_when_no_files(self):
        config = load_config(search_paths=["/nonexistent/path"])
        assert config == {}

    def test_loads_json_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "high", "top_n": 10}, f)
            path = f.name
        try:
            config = load_config(search_paths=[path])
            assert config["model"] == "high"
            assert config["top_n"] == 10
        finally:
            os.unlink(path)

    def test_loads_yaml_config_when_available(self):
        pytest.importorskip("yaml", reason="PyYAML not installed")
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": "ultra", "quiet": True}, f)
            path = f.name
        try:
            config = load_config(search_paths=[path])
            assert config["model"] == "ultra"
            assert config["quiet"] is True
        finally:
            os.unlink(path)

    def test_env_vars_override_file(self, monkeypatch):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "default", "top_n": 10}, f)
            path = f.name
        try:
            monkeypatch.setenv("PATHFINDER_MODEL", "high")
            monkeypatch.setenv("PATHFINDER_TOP_N", "5")
            config = load_config(search_paths=[path])
            assert config["model"] == "high"
            assert config["top_n"] == 5
        finally:
            os.unlink(path)

    def test_kwargs_override_everything(self, monkeypatch):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "default"}, f)
            path = f.name
        try:
            monkeypatch.setenv("PATHFINDER_MODEL", "high")
            config = load_config(search_paths=[path], overrides={"model": "ultra"})
            assert config["model"] == "ultra"
        finally:
            os.unlink(path)

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_config(search_paths=[path])
        finally:
            os.unlink(path)

    def test_bool_env_var_parsing(self, monkeypatch):
        monkeypatch.setenv("PATHFINDER_QUIET", "true")
        config = load_config(search_paths=[])
        assert config["quiet"] is True

    def test_int_env_var_parsing(self, monkeypatch):
        monkeypatch.setenv("PATHFINDER_TOP_N", "15")
        config = load_config(search_paths=[])
        assert config["top_n"] == 15

    def test_float_env_var_parsing(self, monkeypatch):
        monkeypatch.setenv("PATHFINDER_RATE_LIMIT", "2.5")
        config = load_config(search_paths=[])
        assert config["rate_limit"] == 2.5
