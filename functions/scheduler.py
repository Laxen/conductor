import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from integrations.memory import MemoryStore, _format_entry


logger = logging.getLogger(__name__)


class SchedulerFunction:
    """Background task that fires timed reminders at the correct minute."""

    def __init__(self, store: MemoryStore):
        self.store = store

    async def run(self, send_fn: Callable[[str], Awaitable[None]]) -> None:
        logger.info("Scheduler started")
        while True:
            now = datetime.now().astimezone()
            # Sleep until the top of the next minute
            seconds_to_sleep = 60 - now.second - now.microsecond / 1_000_000
            await asyncio.sleep(seconds_to_sleep)

            now = datetime.now().astimezone()
            minute_str = now.strftime("%Y-%m-%dT%H:%M")

            try:
                entries = self.store.get_due_at(minute_str)
            except Exception:
                logger.exception("Scheduler failed to fetch entries for %s", minute_str)
                continue

            if entries:
                lines = [f"• {_format_entry(m)}" for m in entries]
                message = "⏰ Reminder\n" + "\n".join(lines)
                try:
                    await send_fn(message)
                except Exception:
                    logger.exception("Scheduler failed to send reminder for %s", minute_str)
