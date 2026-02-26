import json
import os

import pandas as pd


def save_results(results: list[dict], output_path: str) -> None:
    """Persist classification results as a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {output_path}")


def load_results(path: str) -> list[dict]:
    """Load previously saved classification results."""
    with open(path) as f:
        return json.load(f)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert classification results to a display-ready DataFrame."""
    df = pd.DataFrame(results)

    if "topics" in df.columns:
        df["topics"] = df["topics"].apply(
            lambda x: "No Match" if isinstance(x, list) and len(x) == 0 else x
        )
        df["topics"] = df["topics"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    return df


def print_results(results: list[dict]) -> None:
    """Pretty-print classification results to stdout."""
    for i, result in enumerate(results, 1):
        print(f"Verbatim {i}: {result['verbatim_text']}")
        topics = result.get("topics", [])
        topic_str = ", ".join(topics) if topics else "No Match"
        print(f"Topics    : {topic_str}\n")
