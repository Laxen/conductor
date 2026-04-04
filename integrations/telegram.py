from collections.abc import Callable
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from settings import get_env


logger = logging.getLogger(__name__)


class TelegramIntegration:
    def __init__(self):
        bot_token = get_env("TELEGRAM_BOT_TOKEN")
        whitelist_raw = get_env("TELEGRAM_USER_WHITELIST")
        self.allowed_user_ids = {
            int(uid.strip()) for uid in whitelist_raw.split(",") if uid.strip()
        }

        self.application = ApplicationBuilder().token(bot_token).build()
        self._callback: Callable[[str, str], str | None] | None = None

    def on_message(self, callback: Callable[[str, str], str | None]) -> None:
        self._callback = callback

        async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user is None or update.message is None:
                return

            user_id = update.effective_user.id
            if user_id not in self.allowed_user_ids:
                logger.warning("Rejected message from non-whitelisted user %s", user_id)
                return

            text = update.message.text or ""
            username = update.effective_user.first_name or ""

            reply = callback(text, username)
            if reply:
                await update.message.reply_text(reply)

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    def start(self) -> None:
        logger.info("Starting Telegram bot (polling)")
        self.application.run_polling()
