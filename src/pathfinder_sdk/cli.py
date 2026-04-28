"""Command-line interface for Pathfinder SDK.

Usage:
    python -m pathfinder_sdk rank <url> <task> [options]
"""

import argparse
import sys
from collections.abc import Sequence

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.models import (
    ConfigurationError,
    FetchError,
    ModelLoadError,
    ModelNotFoundError,
    RankingResult,
)


def _format_table(result: RankingResult) -> str:
    """Format a RankingResult as a human-readable table."""
    lines: list[str] = []
    lines.append(f"Task: {result.task_description}")
    lines.append(f"URL:  {result.source_url}")
    lines.append(f"Model: {result.model_tier} | Latency: {result.latency_ms:.2f} ms")
    lines.append(
        f"Links: {result.total_links_analyzed} analyzed, "
        f"{result.total_links_after_filter} after filter"
    )
    lines.append("")

    if not result.candidates:
        lines.append("No candidates found.")
        return "\n".join(lines)

    # Header
    lines.append(f"{'Rank':<6} {'Score':<8} {'Href':<40} {'Text'}")
    lines.append("-" * 80)

    for c in result.candidates:
        text = c.text.replace("\n", " ").replace("\r", " ")[:30]
        href = c.href[:38]
        lines.append(f"{c.rank:<6} {c.score:<8.3f} {href:<40} {text}")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathfinder_sdk",
        description="Local ranking engine for AI navigation agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser(
        "rank", help="Rank candidate links on a page for a given task."
    )
    rank_parser.add_argument("url", help="Target page URL.")
    rank_parser.add_argument(
        "task", help="Natural-language task description (e.g. 'Find the tutorial')."
    )
    rank_parser.add_argument(
        "--model",
        default="default",
        choices=["default", "high", "ultra"],
        help="Model tier to use (default: %(default)s).",
    )
    rank_parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top candidates to return (default: %(default)s).",
    )
    rank_parser.add_argument(
        "--output",
        default="table",
        choices=["json", "table"],
        help="Output format (default: %(default)s).",
    )
    rank_parser.add_argument(
        "--cache-dir",
        default="~/.cache/pathfinder",
        help="Directory for cached model downloads (default: %(default)s).",
    )
    rank_parser.add_argument(
        "--fetcher",
        default="auto",
        choices=["auto", "curl", "playwright"],
        help="Fetcher backend (default: %(default)s).",
    )
    rank_parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default: auto-detect CPU).",
    )
    rank_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output during model download.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 = success, 1 = fetch error, 2 = model error,
        3 = no candidates, 4 = configuration error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "rank":
        parser.print_help()
        return 4

    sdk: Pathfinder | None = None
    try:
        sdk = Pathfinder(
            model=args.model,
            top_n=args.top_n,
            cache_dir=args.cache_dir,
            fetcher=args.fetcher,
            device=args.device,
            quiet=args.quiet,
        )

        result = sdk.rank_candidates(
            url=args.url,
            task_description=args.task,
            top_n=args.top_n,
        )

        if args.output == "json":
            print(result.to_json())
        else:
            print(_format_table(result))

        if not result.candidates:
            return 3

        return 0

    except FetchError as exc:
        print(f"Fetch error: {exc}", file=sys.stderr)
        return 1
    except (ModelNotFoundError, ModelLoadError) as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        return 2
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    finally:
        if sdk is not None:
            sdk.unload()


if __name__ == "__main__":
    sys.exit(main())
