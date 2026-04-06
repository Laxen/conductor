import logging
from collections.abc import Callable

from functions.briefing import BriefingFunction
from integrations.memory import MemoryApp, MemoryStore
from integrations.openai import OpenAIIntegration
from integrations.telegram import TelegramIntegration
from settings import configure_logging, get_env


logger = logging.getLogger(__name__)


def main():
    log_level = get_env("LOG_LEVEL")
    configure_logging(log_level)

    store = MemoryStore()
    store.init()

    openai = OpenAIIntegration()
    telegram = TelegramIntegration()
    app = MemoryApp(store, openai)
    briefing = BriefingFunction(store, openai)

    def on_message(text: str, username: str, confirm_fn: Callable[[str, dict], bool] | None) -> str | None:
        if text.strip().lower() == "brief":
            return briefing.execute()
        return app.handle_input(text, username, confirm_fn)

    telegram.on_message(on_message)
    telegram.on_command("show", "Show memories in the database", app.handle_show)
    telegram.on_command("schedule", "Show upcoming schedule", app.handle_schedule)
    telegram.start()


if __name__ == "__main__":
    main()
