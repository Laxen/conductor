import logging
from collections.abc import Callable, Awaitable

from aiohttp import web

from integrations.memory import MemoryStore, Metadata


logger = logging.getLogger(__name__)


class WebhookIntegration:
    """Runs a small HTTP server so Home Assistant automations can trigger Conductor."""

    def __init__(self, store: MemoryStore, send_fn: Callable[[str], Awaitable[None]], port: int):
        self.store = store
        self.send_fn = send_fn
        self.port = port

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/webhook", self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("Webhook server listening on port %s", self.port)

    async def _handle(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON")

        event = data.get("event")

        if event == "zone_entered":
            return await self._handle_zone_entered(data)

        return web.Response(status=400, text=f"Unknown event: {event!r}")

    async def _handle_zone_entered(self, data: dict) -> web.Response:
        zone = data.get("zone")
        if not zone:
            return web.Response(status=400, text="Missing 'zone'")

        memories = self.store.retrieve_by_metadata(Metadata(location=zone))
        if memories:
            lines = [f"• {m.raw_text}{m.metadata.display()}" for m in memories]
            message = f"📍 {zone}\n" + "\n".join(lines)
            await self.send_fn(message)
        return web.Response(text="OK")
