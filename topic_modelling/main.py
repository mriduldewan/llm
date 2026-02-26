"""
Topic Modelling — entry point.

Usage:
    python main.py --backend openai
    python main.py --backend deepseek
    python main.py --backend gpt-oss
"""

import argparse
import sys

from config.settings import (
    OPENAI_MODEL,
    OLLAMA_MODEL_DEEPSEEK,
    OLLAMA_MODEL_GPT_OSS,
)
from src.utils.io import print_results, results_to_dataframe

# ---------------------------------------------------------------------------
# Sample verbatims (replace with your real data source)
# ---------------------------------------------------------------------------
VERBATIMS = [
    "My classes are all over the place. I have to come to campus three times a week for just one or two hours each time.",
    "The course material is outdated; we're still learning about software from five years ago.",
    "I tried to get help from the student welfare office, but they were closed.",
    "My final assignment feedback was very vague, and I don't know what to improve on.",
    "I'm not sure if this is the right career path for me after finishing this course.",
    "The computer labs have really old computers, and some don't even work properly.",
    "The process to apply for this course was so confusing and the website kept crashing.",
    "My work placement was very unorganized and I felt like I didn't learn anything.",
    "The person who was meant to help me with my enrolment never got back to me.",
    "The cost of the textbooks is way too high, and I'm not sure if I can afford them.",
    "This feedback is not about any of the topics.",
    "I need help with my resume and job applications after I graduate.",
    "The fees for next semester seem to have increased without much warning.",
    "The campus security could be better, I don't feel entirely safe at night.",
    "I really enjoy the practical exercises in this course; they are very relevant to industry.",
]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def build_classifier(backend: str):
    if backend == "openai":
        from src.classifiers.openai_classifier import OpenAIBatchClassifier
        return OpenAIBatchClassifier(model=OPENAI_MODEL)

    if backend == "deepseek":
        from src.classifiers.ollama_classifier import OllamaClassifier
        return OllamaClassifier(
            model=OLLAMA_MODEL_DEEPSEEK,
            json_strategy="json_mode",
        )

    if backend == "gpt-oss":
        from src.classifiers.ollama_classifier import OllamaClassifier
        return OllamaClassifier(
            model=OLLAMA_MODEL_GPT_OSS,
            json_strategy="regex",
        )

    print(f"Unknown backend: '{backend}'. Choose from: openai, deepseek, gpt-oss")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic Modelling for student verbatims")
    parser.add_argument(
        "--backend",
        choices=["openai", "deepseek", "gpt-oss"],
        default="openai",
        help="LLM backend to use (default: openai)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = build_classifier(args.backend)

    print(f"\n--- Running with backend: {args.backend} ---\n")
    results = classifier.classify_batch(VERBATIMS)

    print_results(results)

    df = results_to_dataframe(results)
    print("\n--- Final Results DataFrame ---")
    print(df[["verbatim_text", "topics"]].to_string(index=False))


if __name__ == "__main__":
    main()
