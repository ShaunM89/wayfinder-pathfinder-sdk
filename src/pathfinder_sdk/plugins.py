"""Plugin system for custom fetchers and rankers.

Supports decorator-based registration and entry-point discovery.
"""

import logging
from importlib.metadata import entry_points

from pathfinder_sdk.fetcher import CurlFetcher, PlaywrightFetcher
from pathfinder_sdk.ranker import BiEncoderRanker

logger = logging.getLogger(__name__)

# Built-in registries
_FETCHER_REGISTRY: dict[str, type] = {
    "curl": CurlFetcher,
    "playwright": PlaywrightFetcher,
}

_RANKER_REGISTRY: dict[str, type] = {
    "default": BiEncoderRanker,
    "high": BiEncoderRanker,
    "ultra": BiEncoderRanker,
}


def register_fetcher(name: str):
    """Decorator to register a custom fetcher class.

    Args:
        name: Unique identifier for the fetcher.

    Raises:
        ValueError: If the name is already registered.
    """

    def decorator(cls: type) -> type:
        if name in _FETCHER_REGISTRY:
            raise ValueError(f"Fetcher '{name}' is already registered")
        _FETCHER_REGISTRY[name] = cls
        logger.debug("Registered fetcher plugin: %s", name)
        return cls

    return decorator


def register_ranker(name: str):
    """Decorator to register a custom ranker class.

    Args:
        name: Unique identifier for the ranker.

    Raises:
        ValueError: If the name is already registered.
    """

    def decorator(cls: type) -> type:
        if name in _RANKER_REGISTRY:
            raise ValueError(f"Ranker '{name}' is already registered")
        _RANKER_REGISTRY[name] = cls
        logger.debug("Registered ranker plugin: %s", name)
        return cls

    return decorator


def discover_plugins() -> None:
    """Discover plugins via entry points.

    Looks for entry points in the groups:
    - `pathfinder_sdk.fetchers`
    - `pathfinder_sdk.rankers`
    """
    try:
        eps = entry_points()
        for ep in eps.select(group="pathfinder_sdk.fetchers"):
            try:
                cls = ep.load()
                if ep.name not in _FETCHER_REGISTRY:
                    _FETCHER_REGISTRY[ep.name] = cls
                    logger.info("Discovered fetcher plugin: %s", ep.name)
            except Exception as exc:
                logger.warning("Failed to load fetcher plugin %s: %s", ep.name, exc)

        for ep in eps.select(group="pathfinder_sdk.rankers"):
            try:
                cls = ep.load()
                if ep.name not in _RANKER_REGISTRY:
                    _RANKER_REGISTRY[ep.name] = cls
                    logger.info("Discovered ranker plugin: %s", ep.name)
            except Exception as exc:
                logger.warning("Failed to load ranker plugin %s: %s", ep.name, exc)
    except Exception as exc:
        logger.debug("Entry point discovery failed: %s", exc)


def resolve_fetcher(name: str) -> type:
    """Resolve a fetcher by name.

    Args:
        name: Fetcher identifier.

    Returns:
        Fetcher class.

    Raises:
        ValueError: If the fetcher is not found.
    """
    if name not in _FETCHER_REGISTRY:
        raise ValueError(
            f"Unknown fetcher backend: {name}. "
            f"Available: {list(_FETCHER_REGISTRY.keys())}"
        )
    return _FETCHER_REGISTRY[name]


def resolve_ranker(name: str) -> type:
    """Resolve a ranker by name.

    Args:
        name: Ranker identifier.

    Returns:
        Ranker class.

    Raises:
        ValueError: If the ranker is not found.
    """
    if name not in _RANKER_REGISTRY:
        raise ValueError(
            f"Unknown ranker: {name}. " f"Available: {list(_RANKER_REGISTRY.keys())}"
        )
    return _RANKER_REGISTRY[name]


# Auto-discover on import
discover_plugins()
