#!/usr/bin/env python3
"""
Point prediction test — summarise a single URL.

Run from the web_summariser/ directory:

    # OpenAI (default)
    python tests/test_point.py

    # Custom URL
    python tests/test_point.py --url https://techcrunch.com

    # Use Ollama instead
    python tests/test_point.py --url https://bbc.com --provider ollama

    # Save output to output/
    python tests/test_point.py --url https://openai.com --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure the package root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=True)

from src.summariser import get_summariser  # noqa: E402


def _load_config() -> dict:
    with open(_ROOT / "config.json") as fh:
        return json.load(fh)


def run(url: str, provider: str, save: bool = False) -> str:
    config = _load_config()
    summariser = get_summariser(provider, config)

    print(f"Provider : {provider}")
    print(f"URL      : {url}")
    print("-" * 60)

    summary = summariser.summarise(url)
    print(summary)

    if save:
        output_dir = _ROOT / config["output"]["directory"]
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
        print(f"\nSaved → {path}")

    return summary


def main() -> None:
    config = _load_config()
    default_provider = config.get("default_provider", "openai")
    valid_providers = list(config["models"].keys())

    parser = argparse.ArgumentParser(
        description="Point prediction: summarise a single URL.",
    )
    parser.add_argument(
        "--url",
        default="https://cnn.com",
        help="URL to summarise (default: https://cnn.com)",
    )
    parser.add_argument(
        "--provider",
        choices=valid_providers,
        default=default_provider,
        help=f"LLM backend. Choices: {valid_providers}. Default: {default_provider}",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the summary to the output/ folder.",
    )
    args = parser.parse_args()
    run(args.url, args.provider, args.save)


if __name__ == "__main__":
    main()
