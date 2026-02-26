from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract base class for all topic classifiers."""

    @abstractmethod
    def classify_batch(self, verbatims: list[str]) -> list[dict]:
        """
        Classify a list of verbatims into topics.

        Args:
            verbatims: List of raw student feedback strings.

        Returns:
            List of dicts with keys: 'custom_id', 'verbatim_text', 'topics'.
        """
        ...
