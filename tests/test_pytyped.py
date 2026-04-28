"""Verify py.typed marker is present for type-checker support."""

import pathfinder_sdk


def test_pytyped_marker_exists():
    """py.typed must be present so mypy/pyright treat package as typed."""
    import importlib.resources as resources

    pkg = resources.files(pathfinder_sdk)
    marker = pkg / "py.typed"
    assert marker.exists(), "py.typed marker is missing from the package"
