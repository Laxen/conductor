from collections.abc import Callable
import html as _html
import logging
import re

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from settings import get_env


logger = logging.getLogger(__name__)


def _md_to_html(text: str) -> str:
    """Convert standard markdown to Telegram-compatible HTML."""
    parts = []
    last_end = 0
    code_pattern = re.compile(r'```(?:\w+)?\n?([\s\S]*?)```|`([^`\n]+)`')
    for match in code_pattern.finditer(text):
        parts.append(_convert_inline(text[last_end:match.start()]))
        if match.group(1) is not None:
            parts.append(f'<pre>{_html.escape(match.group(1).strip())}</pre>')
        else:
            parts.append(f'<code>{_html.escape(match.group(2))}</code>')
        last_end = match.end()
    parts.append(_convert_inline(text[last_end:]))
    return ''.join(parts)


def _convert_inline(text: str) -> str:
    """Convert non-code markdown formatting to Telegram-compatible HTML tags."""
    text = _html.escape(text)
    text = re.sub(r'\*\*([^\n]+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^\n]+?)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<![_\w])_([^_\n]+)_(?![_\w])', r'<i>\1</i>', text)
    text = re.sub(r'~~([^\n]+?)~~', r'<s>\1</s>', text)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    return text


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
                await update.message.reply_text(_md_to_html(reply), parse_mode=ParseMode.HTML)

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    def on_command(self, command: str, description: str, callback: Callable[[list[str]], str | None]) -> None:
        async def _handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user is None or update.message is None:
                return

            user_id = update.effective_user.id
            if user_id not in self.allowed_user_ids:
                logger.warning("Rejected command from non-whitelisted user %s", user_id)
                return

            reply = callback(list(context.args or []))
            if reply:
                await update.message.reply_text(_md_to_html(reply), parse_mode=ParseMode.HTML)

        self.application.add_handler(CommandHandler(command, _handle_command))
        self._commands.append(BotCommand(command, description))

    def start(self) -> None:
        logger.info("Starting Telegram bot (polling)")

        async def _post_init(app):
            await app.bot.set_my_commands(self._commands)
            for user_id in self.allowed_user_ids:
                await app.bot.send_message(chat_id=user_id, text="Conductor started")

        self.application.post_init = _post_init
        self.application.run_polling()
