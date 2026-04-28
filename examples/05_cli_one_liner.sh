#!/usr/bin/env bash
# Example 5: CLI one-liners using python -m pathfinder_sdk
#
# Run:
#     chmod +x examples/05_cli_one_liner.sh
#     ./examples/05_cli_one_liner.sh

set -e

echo "=== Example 5: CLI one-liners ==="
echo ""

# Basic usage: rank candidates on a page
echo "1. Basic ranking (table output):"
python -m pathfinder_sdk rank "https://docs.python.org/3/" "Find the tutorial" \
    --fetcher none \
    2>/dev/null || echo "   (Skipped: requires network or pre-extracted candidates)"

echo ""
echo "2. JSON output for piping to jq or other tools:"
python -m pathfinder_sdk rank "https://docs.python.org/3/" "Find the tutorial" \
    --output json \
    --fetcher none \
    2>/dev/null || echo "   (Skipped: requires network or pre-extracted candidates)"

echo ""
echo "3. Using a higher-quality model:"
python -m pathfinder_sdk rank "https://docs.python.org/3/" "Find the tutorial" \
    --model high \
    --top-n 5 \
    --fetcher none \
    2>/dev/null || echo "   (Skipped: requires network or pre-extracted candidates)"

echo ""
echo "4. Check exit codes:"
python -m pathfinder_sdk rank --help

echo ""
echo "Done."
