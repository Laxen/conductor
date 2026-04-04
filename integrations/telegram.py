from collections.abc import Callable
import logging

from telegram import BotCommand, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
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
        self._commands: list[BotCommand] = []

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

    def on_command(self, command: str, description: str, callback: Callable[[], str | None]) -> None:
        async def _handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user is None or update.message is None:
                return

            user_id = update.effective_user.id
            if user_id not in self.allowed_user_ids:
                logger.warning("Rejected command from non-whitelisted user %s", user_id)
                return

            reply = callback()
            if reply:
                await update.message.reply_text(reply)

        self.application.add_handler(CommandHandler(command, _handle_command))
        self._commands.append(BotCommand(command, description))

    def start(self) -> None:
        logger.info("Starting Telegram bot (polling)")

        async def _post_init(app):
            await app.bot.set_my_commands(self._commands)

        self.application.post_init = _post_init
        self.application.run_polling()
