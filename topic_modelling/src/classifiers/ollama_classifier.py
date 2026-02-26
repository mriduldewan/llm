import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ollama import Client

from src.classifiers.base import BaseClassifier
from src.prompts import messages_for
from config.settings import OLLAMA_HOST, MAX_CONCURRENT_REQUESTS


class OllamaClassifier(BaseClassifier):
    """
    Classifies student verbatims using a locally running Ollama model.

    Two JSON extraction strategies are supported:
    - "json_mode": passes format="json" to Ollama (works with models that
      support native JSON mode, e.g. deepseek-r1:8b).
    - "regex": extracts the first JSON object from the raw text response
      (fallback for models that ignore the format flag, e.g. gpt-oss:20b).
    """

    def __init__(
        self,
        model: str,
        json_strategy: str = "json_mode",
        max_workers: int = MAX_CONCURRENT_REQUESTS,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_predict: int = 200,
    ):
        if json_strategy not in ("json_mode", "regex"):
            raise ValueError("json_strategy must be 'json_mode' or 'regex'")
        self.client = Client(host=OLLAMA_HOST)
        self.model = model
        self.json_strategy = json_strategy
        self.max_workers = max_workers
        self.ollama_options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify_batch(self, verbatims: list[str]) -> list[dict]:
        items = [
            {"custom_id": f"verbatim_{i + 1}", "verbatim_text": v}
            for i, v in enumerate(verbatims)
        ]

        print(f"Starting batch processing with {self.max_workers} concurrent workers "
              f"[model={self.model}, strategy={self.json_strategy}]...")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._classify_one, item): item for item in items}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                print(f"  Processed {len(results)}/{len(items)}...", end="\r")

        print(f"\nBatch processing complete. {len(results)} results returned.")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_one(self, item: dict) -> dict | None:
        verbatim = item["verbatim_text"]
        custom_id = item["custom_id"]

        try:
            kwargs = dict(
                model=self.model,
                messages=messages_for(verbatim),
                options=self.ollama_options,
            )
            if self.json_strategy == "json_mode":
                kwargs["format"] = "json"

            response = self.client.chat(**kwargs)
            raw = response["message"]["content"]

            classification = self._parse_response(raw, custom_id, verbatim)
            return classification

        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for {custom_id}: {e}")
            return {"custom_id": custom_id, "verbatim_text": verbatim, "topics": ["Error: JSON Decode Error"]}
        except Exception as e:
            print(f"\nError processing {custom_id}: {e}")
            return {"custom_id": custom_id, "verbatim_text": verbatim, "topics": ["Error: API Call Failed"]}

    def _parse_response(self, raw: str, custom_id: str, verbatim: str) -> dict:
        if self.json_strategy == "json_mode":
            data = json.loads(raw)
        else:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                print(f"\nWarning: no JSON found for {custom_id}.")
                return {"custom_id": custom_id, "verbatim_text": verbatim, "topics": ["No Match"]}
            data = json.loads(match.group(0))

        return {
            "custom_id": data.get("custom_id", custom_id),
            "verbatim_text": data.get("verbatim_text", verbatim),
            "topics": data.get("topics", ["Error: No topics"]),
        }
