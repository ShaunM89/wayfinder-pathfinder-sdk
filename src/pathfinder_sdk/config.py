"""Configuration file support for Pathfinder SDK.

Search path (in order of priority):
1. Explicit path passed to load_config()
2. ./.pathfinder.yaml (project-local)
3. ~/.config/pathfinder/config.yaml (user-global)
4. Environment variables (PATHFINDER_*)
5. Constructor kwargs (highest priority, passed as overrides)

YAML is preferred; JSON is used as fallback. If PyYAML is not installed,
only JSON files are supported.
"""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Config keys that map to PATHFINDER_* env vars
_ENV_PREFIX = "PATHFINDER_"

# Default search paths
_DEFAULT_SEARCH_PATHS = [
    "./.pathfinder.yaml",
    "./.pathfinder.json",
    "~/.config/pathfinder/config.yaml",
    "~/.config/pathfinder/config.json",
]

# Type conversion for env vars
_ENV_TYPE_MAP: dict[str, Callable[[str], object]] = {
    "top_n": int,
    "max_retries": int,
    "max_requests_per_domain": int,
    "quiet": lambda v: v.lower() in ("true", "1", "yes"),
    "rate_limit": float,
    "retry_delay": float,
    "timeout": int,
}


def _load_yaml(path: str) -> dict:
    """Load YAML config file."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        ) from exc


def _load_json(path: str) -> dict:
    """Load JSON config file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc


def _load_file(path: str) -> dict:
    """Load config from file (YAML or JSON)."""
    path = str(Path(path).expanduser())
    if not os.path.isfile(path):
        return {}

    if path.endswith((".yaml", ".yml")):
        return _load_yaml(path)
    elif path.endswith(".json"):
        return _load_json(path)
    else:
        # Try YAML first, then JSON
        try:
            return _load_yaml(path)
        except ImportError:
            return _load_json(path)


def _load_env_vars() -> dict:
    """Load configuration from PATHFINDER_* environment variables."""
    config: dict = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            config_key = key[len(_ENV_PREFIX) :].lower()
            # Type conversion
            converter = _ENV_TYPE_MAP.get(config_key)
            if converter:
                try:
                    config[config_key] = converter(value)
                except (ValueError, TypeError):
                    config[config_key] = value
            else:
                config[config_key] = value
    return config


def load_config(
    path: str | None = None,
    search_paths: list[str] | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load Pathfinder configuration from files and environment.

    Priority (lowest to highest):
    1. Config file (first found in search_paths)
    2. Environment variables (PATHFINDER_*)
    3. Explicit overrides kwargs

    Args:
        path: Explicit config file path. If provided, skips search_paths.
        search_paths: List of paths to search for config files.
        overrides: Dictionary of explicit overrides (highest priority).

    Returns:
        Merged configuration dictionary.
    """
    config: dict = {}

    # 1. Load from file
    if path is not None:
        config.update(_load_file(path))
    else:
        paths = search_paths if search_paths is not None else _DEFAULT_SEARCH_PATHS
        for p in paths:
            file_config = _load_file(p)
            if file_config:
                config.update(file_config)
                logger.debug("Loaded config from %s", p)
                break

    # 2. Environment variables override file
    config.update(_load_env_vars())

    # 3. Explicit overrides (highest priority)
    if overrides:
        config.update(overrides)

    return config
