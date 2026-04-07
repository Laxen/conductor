import logging
import os


PROMPT_PATH = "/data/prompt.txt"
logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "You are an assistant managing a memory-base for a user, with entries being tasks, events or general information. "
    "Use the available tools to fully understand the user's intention and fulfill their request. "
    "If the user states a fact/task/reminder/event, check if there's already a closely matching entry that can be updated with that info, otherwise add a new one. "
    "If the user says something is done/finished or checked off, get the entry they're talking about and delete it. "
    "NEVER add extra text to an entry that the user didn't input. NEVER add metadata (date, location, tag, etc.) unless it's obvious from the user's message. "
    "NEVER lie about what you've done or make things up, if you don't understand the user's request then tell them why. "
    "Any relative date phrases like 'tomorrow' and weekdays like 'on Thursday' should be calculated deterministically from the given reference date. NEVER guess. "
    "When you have finished all necessary actions, respond with a summary of what you've done."
)


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
