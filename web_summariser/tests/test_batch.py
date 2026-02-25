#!/usr/bin/env python3
"""
Batch prediction test — summarise multiple URLs from a JSON file.

The input JSON file must be either:
  - A list of URL strings:       ["https://cnn.com", "https://bbc.com"]
  - A list of objects with 'url': [{"url": "https://cnn.com", "label": "CNN"}, …]

Run from the web_summariser/ directory:

    # Use default batch file and OpenAI
    python tests/test_batch.py

    # Custom input file
    python tests/test_batch.py --input tests/data/batch_urls.json

    # Use Ollama and save results
    python tests/test_batch.py --provider ollama --save

    # Save results to output/ (individual .md files + batch_report_*.json)
    python tests/test_batch.py --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=True)

from src.summariser import get_summariser  # noqa: E402


def _load_config() -> dict:
    with open(_ROOT / "config.json") as fh:
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


def run(input_file: str, provider: str, save: bool = False) -> list[dict]:
    config = _load_config()
    input_path = Path(input_file)

    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    with open(input_path) as fh:
        raw = json.load(fh)

    urls: list[str] = [
        item if isinstance(item, str) else item["url"] for item in raw
    ]

    if not urls:
        sys.exit("Error: no URLs found in input file.")

    summariser = get_summariser(provider, config)
    output_dir = _ROOT / config["output"]["directory"]

    print(f"Provider : {provider}")
    print(f"Input    : {input_path}")
    print(f"URLs     : {len(urls)}")
    print("=" * 60)

    results: list[dict] = []

    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] {url}")
        print("-" * 60)
        try:
            summary = summariser.summarise(url)
            results.append({"url": url, "summary": summary, "status": "success"})
            print(summary)
            if save:
                path = _save_summary(url, summary, output_dir)
                print(f"\nSaved → {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            results.append({"url": url, "error": str(exc), "status": "failed"})

    # Final stats
    success_count = sum(1 for r in results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{len(urls)} succeeded.")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"batch_report_{timestamp}.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Batch report → {report_path}")

    return results


def main() -> None:
    config = _load_config()
    default_provider = config.get("default_provider", "openai")
    valid_providers = list(config["models"].keys())
    default_input = str(_ROOT / "tests" / "data" / "batch_urls.json")

    parser = argparse.ArgumentParser(
        description="Batch prediction: summarise multiple URLs from a JSON file.",
    )
    parser.add_argument(
        "--input",
        default=default_input,
        metavar="FILE",
        help=f"JSON file containing URLs to summarise (default: {default_input})",
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
        help="Write each summary and a batch_report.json to the output/ folder.",
    )
    args = parser.parse_args()
    run(args.input, args.provider, args.save)


if __name__ == "__main__":
    main()
