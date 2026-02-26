import json
import time
import os

from openai import OpenAI

from src.classifiers.base import BaseClassifier
from src.prompts import SYSTEM_PROMPT, user_prompt_for
from config.settings import OPENAI_MODEL, INPUT_DIR, OUTPUT_DIR


class OpenAIBatchClassifier(BaseClassifier):
    """
    Classifies student verbatims using the OpenAI Batch API.
    Submits a JSONL batch job, polls until complete, then downloads results.
    """

    def __init__(self, model: str = OPENAI_MODEL, poll_interval: int = 30):
        self.client = OpenAI()
        self.model = model
        self.poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify_batch(self, verbatims: list[str]) -> list[dict]:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"batch_in_{timestamp}.jsonl"
        output_filename = f"batch_out_{timestamp}.json"

        input_path = self._create_batch_input(verbatims, input_filename)
        input_file_id = self._upload_file(input_path)
        batch_job_id = self._create_batch_job(input_file_id)
        retrieved_job, status = self._poll_until_done(batch_job_id)
        results = self._download_results(retrieved_job, status, output_filename)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_batch_input(self, verbatims: list[str], filename: str) -> str:
        os.makedirs(INPUT_DIR, exist_ok=True)
        path = os.path.join(INPUT_DIR, filename)

        with open(path, "w") as f:
            for i, verbatim in enumerate(verbatims):
                request_body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt_for(verbatim)},
                    ],
                    "response_format": {"type": "json_object"},
                }
                batch_request = {
                    "custom_id": f"verbatim_{i + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body,
                }
                f.write(json.dumps(batch_request) + "\n")

        print(f"Batch input file created: {path}")
        return path

    def _upload_file(self, path: str) -> str:
        batch_input_file = self.client.files.create(
            file=open(path, "rb"), purpose="batch"
        )
        print(f"File uploaded with ID: {batch_input_file.id}")
        return batch_input_file.id

    def _create_batch_job(self, input_file_id: str) -> str:
        batch_job = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Batch job created with ID: {batch_job.id}")
        return batch_job.id

    def _poll_until_done(self, batch_job_id: str):
        print(f"Polling batch job {batch_job_id}...")
        attempt = 1
        while True:
            job = self.client.batches.retrieve(batch_job_id)
            status = job.status
            print(f"  Attempt {attempt}: {status}")
            if status in ("completed", "failed", "cancelled", "expired"):
                return job, status
            time.sleep(self.poll_interval)
            attempt += 1

    def _download_results(self, retrieved_job, status: str, output_filename: str) -> list[dict]:
        if status != "completed":
            print(f"Batch job ended with status: {status}")
            if retrieved_job.error_file_id:
                errors = self.client.files.content(retrieved_job.error_file_id).text
                print("--- Error Log ---\n", errors)
            return []

        output_content = self.client.files.content(retrieved_job.output_file_id).text
        parsed_results = []

        for line in output_content.strip().split("\n"):
            try:
                batch_result = json.loads(line)
                custom_id = batch_result.get("custom_id")
                response_content = batch_result["response"]["body"]["choices"][0]["message"]["content"]
                classification_data = json.loads(response_content)
                parsed_results.append(
                    {
                        "custom_id": custom_id,
                        "verbatim_text": classification_data["verbatim_text"],
                        "topics": classification_data["topics"],
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line: {e}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w") as f:
            json.dump(parsed_results, f, indent=4)

        print(f"Results saved to: {output_path}")
        return parsed_results
