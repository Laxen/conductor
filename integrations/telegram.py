from collections.abc import Callable
import asyncio
import html as _html
import logging
import re
import threading

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, MessageHandler, filters, ContextTypes
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
        self.chat_id = int(get_env("TELEGRAM_CHAT_ID"))

        self.application = ApplicationBuilder().token(bot_token).concurrent_updates(True).build()
        self._callback: Callable[[str, Callable[[str, dict], bool]], str | None] | None = None
        self._commands: list[BotCommand] = []
        self._confirmation_event: threading.Event | None = None
        self._confirmation_result: bool = False

        self._setup_confirmation_handler()

    def _setup_confirmation_handler(self) -> None:
        async def _handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            if query is None or not query.data:
                return
            await query.answer()

            accepted = query.data == "confirm:accept"
            self._confirmation_result = accepted
            if self._confirmation_event is not None:
                self._confirmation_event.set()

            label = "✅ Accepted" if accepted else "❌ Cancelled"
            if query.message:
                try:
                    await query.edit_message_text(
                        query.message.text_html + f"\n\n{label}",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception:
                    logger.exception("Failed to edit confirmation message")

        self.application.add_handler(CallbackQueryHandler(_handle_callback_query))

    async def _send_confirmation_message(self, tool_name: str, args: dict) -> None:
        params_lines = "\n".join(
            f"  <b>{_html.escape(k)}</b>: {_html.escape(str(v))}" for k, v in args.items()
        )
        text = f"⚠️ Confirm <b>{_html.escape(tool_name)}</b>\n\n{params_lines}"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Accept", callback_data="confirm:accept"),
                InlineKeyboardButton("❌ Cancel", callback_data="confirm:cancel"),
            ]
        ])
        await self.application.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )

    def _make_confirm_fn(self, loop: asyncio.AbstractEventLoop) -> Callable[[str, dict], bool]:
        def confirm(tool_name: str, args: dict) -> bool:
            self._confirmation_event = threading.Event()
            self._confirmation_result = False

            asyncio.run_coroutine_threadsafe(
                self._send_confirmation_message(tool_name, args),
                loop,
            ).result()

            self._confirmation_event.wait()
            return self._confirmation_result

        return confirm

    def on_message(self, callback: Callable[[str, Callable[[str, dict], bool]], str | None]) -> None:
        self._callback = callback

        async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user is None or update.message is None:
                return

            if update.effective_user.id != self.chat_id:
                logger.warning("Rejected message from unexpected user %s", update.effective_user.id)
                return

            text = update.message.text or ""
            loop = asyncio.get_running_loop()
            confirm_fn = self._make_confirm_fn(loop)
            reply = await asyncio.to_thread(callback, text, confirm_fn)
            if reply:
                await update.message.reply_text(_md_to_html(reply), parse_mode=ParseMode.HTML)

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    def on_command(self, command: str, description: str, callback: Callable[[list[str]], str | None]) -> None:
        async def _handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user is None or update.message is None:
                return

            if update.effective_user.id != self.chat_id:
                logger.warning("Rejected command from unexpected user %s", update.effective_user.id)
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
            await app.bot.send_message(chat_id=self.chat_id, text="Conductor started")

        self.application.post_init = _post_init
        self.application.run_polling()
