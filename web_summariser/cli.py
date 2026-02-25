#!/usr/bin/env python3
"""
web_summariser CLI

Usage examples
--------------
# Single URL with OpenAI (default provider)
python cli.py single https://cnn.com

# Single URL with Ollama, save to output/
python cli.py --provider ollama single https://bbc.com --save

# Batch from a JSON file with OpenAI
python cli.py batch tests/data/batch_urls.json

# Batch with Ollama, save all summaries and a report
python cli.py --provider ollama batch tests/data/batch_urls.json --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Explicitly load the .env that lives next to this file so the correct key
# is always picked up regardless of which directory the CLI is run from.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent


def _load_config(config_path: Path | str = _ROOT / "config.json") -> dict:
    with open(config_path) as fh:
        return json.load(fh)


def _save_summary(url: str, summary: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = (
        url.removeprefix("https://")
        .removeprefix("http://")
        .replace("/", "_")
        .replace(".", "_")[:60]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{safe}_{timestamp}.md"
    path.write_text(f"# Summary: {url}\n\n{summary}", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_single(args: argparse.Namespace, config: dict) -> None:
    from src.summariser import get_summariser

    summariser = get_summariser(args.provider, config)
    print(f"Provider : {args.provider}")
    print(f"URL      : {args.url}")
    print("-" * 60)

    summary = summariser.summarise(args.url)
    print(summary)

    if args.save:
        output_dir = _ROOT / config["output"]["directory"]
        path = _save_summary(args.url, summary, output_dir)
        print(f"\nSaved → {path}")


def _cmd_batch(args: argparse.Namespace, config: dict) -> None:
    from src.summariser import get_summariser

    input_path = Path(args.input_file)
    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    with open(input_path) as fh:
        raw = json.load(fh)

    # Accept a plain list of URL strings OR a list of {"url": …} objects
    urls: list[str] = [
        item if isinstance(item, str) else item["url"] for item in raw
    ]

    summariser = get_summariser(args.provider, config)
    print(f"Provider : {args.provider}")
    print(f"URLs     : {len(urls)}")
    print("=" * 60)

    results: list[dict] = []
    output_dir = _ROOT / config["output"]["directory"]

    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] {url}")
        print("-" * 60)
        try:
            summary = summariser.summarise(url)
            results.append({"url": url, "summary": summary, "status": "success"})
            print(summary)
            if args.save:
                path = _save_summary(url, summary, output_dir)
                print(f"\nSaved → {path}")
        except Exception as exc:  # noqa: BLE001
            msg = f"ERROR: {exc}"
            print(msg, file=sys.stderr)
            results.append({"url": url, "error": str(exc), "status": "failed"})

    # Summary line
    success_count = sum(1 for r in results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"Done: {success_count}/{len(urls)} succeeded.")

    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"batch_report_{timestamp}.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Batch report → {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser(config: dict) -> argparse.ArgumentParser:
    default_provider = config.get("default_provider", "openai")
    valid_providers = list(config["models"].keys())

    parser = argparse.ArgumentParser(
        prog="web_summariser",
        description="Summarise web pages using closed (OpenAI) or open-source (Ollama) LLMs.",
    )
    parser.add_argument(
        "--provider",
        choices=valid_providers,
        default=default_provider,
        metavar="PROVIDER",
        help=f"LLM backend to use. Choices: {valid_providers}. Default: {default_provider}",
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "config.json"),
        help="Path to config.json (default: ./config.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- single -----------------------------------------------------------
    single = sub.add_parser("single", help="Summarise a single URL.")
    single.add_argument("url", help="The URL to summarise.")
    single.add_argument(
        "--save", action="store_true", help="Write the summary to the output/ folder."
    )

    # -- batch ------------------------------------------------------------
    batch = sub.add_parser(
        "batch", help="Summarise multiple URLs from a JSON file."
    )
    batch.add_argument(
        "input_file",
        help="Path to a JSON file containing a list of URLs (strings) or objects with a 'url' key.",
    )
    batch.add_argument(
        "--save",
        action="store_true",
        help="Write each summary and a batch report to the output/ folder.",
    )

    return parser


def main() -> None:
    # Load config first so the parser can reflect dynamic choices
    config = _load_config()
    parser = _build_parser(config)
    args = parser.parse_args()

    # Allow --config override to reload
    if args.config != str(_ROOT / "config.json"):
        config = _load_config(args.config)

    if args.command == "single":
        _cmd_single(args, config)
    elif args.command == "batch":
        _cmd_batch(args, config)


if __name__ == "__main__":
    main()
