import logging

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

    def on_message(text: str, username: str) -> str | None:
        if text.strip().lower() == "brief":
            return briefing.execute()
        return app.handle_input(text, username)

    telegram.on_message(on_message)
    telegram.on_command("showdb", "Show all memories in the database", app.handle_showdb)
    telegram.start()


if __name__ == "__main__":
    main()
