import json
import logging

OPTIONS_PATH = "/data/options.json"

APP_LOGGERS = ("__main__", "main", "settings", "integrations")


def _load_options() -> dict:
    try:
        with open(OPTIONS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_options = _load_options()


def configure_logging(log_level: str = "INFO") -> None:
    """Set up logging so only this project's modules use *log_level*.

    Third-party packages stay at WARNING.
    """
    level_name = (log_level or "WARNING").upper()
    app_level = getattr(logging, level_name, logging.WARNING)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(app_level)


def get_env(name: str) -> str:
    value = _options.get(name.lower())
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required configuration option: {name}")
    return str(value)
