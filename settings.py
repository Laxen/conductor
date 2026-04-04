import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

APP_LOGGERS = ("__main__", "main", "settings", "integrations")


def configure_logging(log_level: str = "DEBUG") -> None:
    """Set up logging so only this project's modules use *log_level*.

    Third-party packages stay at WARNING.
    """
    level_name = (log_level or "WARNING").upper()
    app_level = getattr(logging, level_name, logging.WARNING)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(app_level)


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value
