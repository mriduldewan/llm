from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod

import requests
from openai import OpenAI

from .scraper import Website
from .prompts import messages_for


class BaseSummariser(ABC):
    """Abstract base class for all summariser backends."""

    @abstractmethod
    def summarise(self, url: str) -> str:
        """Fetch the page at *url* and return a markdown summary."""
        ...


class OpenAISummariser(BaseSummariser):
    """Uses the OpenAI chat-completions API (GPT models)."""

    def __init__(self, model: str = "gpt-4o-mini", scraper_config: dict | None = None):
        self.model = model
        self._scraper_cfg = scraper_config or {}
        self._client = OpenAI()

    def summarise(self, url: str) -> str:
        website = Website(url, **self._scraper_cfg)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages_for(website),
        )
        return response.choices[0].message.content


class OllamaSummariser(BaseSummariser):
    """Uses a locally running Ollama instance via its REST API."""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        api_base: str = "http://localhost:11434/api/chat",
        scraper_config: dict | None = None,
        auto_pull: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self._scraper_cfg = scraper_config or {}
        self._headers = {"Content-Type": "application/json"}

        if auto_pull:
            self._ensure_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Pull the model if it is not already present in Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            if self.model not in result.stdout:
                print(f"Model '{self.model}' not found locally — pulling…")
                subprocess.run(["ollama", "pull", self.model], check=True)
                print(f"Model '{self.model}' downloaded successfully.")
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama binary not found. "
                "Install Ollama from https://ollama.com and ensure it is in PATH."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarise(self, url: str) -> str:
        website = Website(url, **self._scraper_cfg)
        payload = {
            "model": self.model,
            "messages": messages_for(website),
            "stream": False,
        }
        response = requests.post(self.api_base, json=payload, headers=self._headers, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_summariser(provider: str, config: dict) -> BaseSummariser:
    """Return the correct BaseSummariser subclass for *provider*.

    Args:
        provider:  One of ``"openai"`` or ``"ollama"``.
        config:    The parsed ``config.json`` dict.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    if provider not in config["models"]:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Valid choices: {list(config['models'].keys())}"
        )

    model_cfg = config["models"][provider]
    scraper_cfg: dict = {}
    if "scraper" in config:
        if "user_agent" in config["scraper"]:
            scraper_cfg["user_agent"] = config["scraper"]["user_agent"]
        if "timeout" in config["scraper"]:
            scraper_cfg["timeout"] = config["scraper"]["timeout"]

    if provider == "openai":
        return OpenAISummariser(model=model_cfg["model"], scraper_config=scraper_cfg)

    if provider == "ollama":
        return OllamaSummariser(
            model=model_cfg["model"],
            api_base=model_cfg["api_base"],
            scraper_config=scraper_cfg,
        )

    raise ValueError(f"Provider '{provider}' is defined in config but has no implementation.")
