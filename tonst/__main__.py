"""
Command-line entry point.

    tonst stats [--log PATH] [--app NAME] [--since YYYY-MM-DD] [--json]

Also runnable without installing the console script:

    python -m tonst stats
"""

from __future__ import annotations
import argparse
import json
import sys

from .savings_log import summarize, format_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tonst", description="tonst command-line tools")
    sub = parser.add_subparsers(dest="command")

    stats = sub.add_parser("stats", help="summarize the local savings log")
    stats.add_argument("--log", default=None, help="log path (default: $TONST_SAVINGS_LOG or ~/.tonst/savings.jsonl)")
    stats.add_argument("--app", default=None, help="only include entries for this app name")
    stats.add_argument("--since", default=None, help="only include entries on/after this ISO date, e.g. 2026-09-01")
    stats.add_argument("--json", action="store_true", help="print machine-readable JSON")

    args = parser.parse_args(argv)
    if args.command != "stats":
        parser.print_help()
        return 1

    summary = summarize(args.log, app=args.app, since=args.since)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
