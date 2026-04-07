import logging
import os

from integrations.openai import DEFAULT_INSTRUCTIONS


PROMPT_PATH = "/data/prompt.txt"
logger = logging.getLogger(__name__)


class PromptStore:
    """Persists a custom system prompt to file, falling back to the default."""

    def __init__(self, path: str = PROMPT_PATH):
        if not os.path.exists(os.path.dirname(path) or "."):
            logger.warning("Prompt data directory not found, saving to current directory instead of %s", path)
            path = "prompt.txt"
        self._path = path

    def load(self) -> str:
        try:
            with open(self._path) as f:
                content = f.read().strip()
                return content if content else DEFAULT_INSTRUCTIONS
        except FileNotFoundError:
            return DEFAULT_INSTRUCTIONS

    def save(self, prompt: str) -> None:
        with open(self._path, "w") as f:
            f.write(prompt)
        logger.info("Prompt saved to %s", self._path)

    def reset(self) -> None:
        try:
            os.remove(self._path)
            logger.info("Prompt reset to default")
        except FileNotFoundError:
            pass
