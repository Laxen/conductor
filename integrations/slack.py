from collections.abc import Callable
import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from settings import get_env


logger = logging.getLogger(__name__)


class SlackIntegration:
    def __init__(self):
        bot_token = get_env("SLACK_BOT_TOKEN")
        app_token = get_env("SLACK_APP_TOKEN")

        self.app = App(token=bot_token)
        self.socket_handler = SocketModeHandler(self.app, app_token)

    def on_dm(self, callback: Callable[[str, str, str], str | None]) -> None:
        @self.app.event("message")
        def _handle_dm(event, say):
            if event.get("channel_type") != "im":
                return
            if event.get("bot_id"):
                return

            text = str(event.get("text", ""))
            channel = str(event.get("channel", ""))
            user_id = str(event.get("user", ""))

            try:
                user_info = self.app.client.users_info(user=user_id)
                profile = user_info["user"]["profile"]
                username = profile.get("display_name") or user_info["user"].get("real_name", "")
            except Exception:
                logger.exception("Failed to resolve Slack user info for %s", user_id)
                username = ""

            reply = callback(text, channel, username)
            if reply:
                say(reply)

    def start(self) -> None:
        self.socket_handler.start()
