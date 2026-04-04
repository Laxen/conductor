import logging
from datetime import datetime, timedelta

from integrations.memory import MemoryStore
from integrations.openai import OpenAIIntegration


logger = logging.getLogger(__name__)

LOOKAHEAD_DAYS = 2


class BriefingFunction:
    def __init__(self, store: MemoryStore, openai: OpenAIIntegration):
        self.store = store
        self.openai = openai

    def execute(self) -> str:
        try:
            memories = self._collect_memories()
            if not memories:
                return "Nothing scheduled for today or the next 2 days."
            return self._summarize(memories)
        except Exception:
            logger.exception("Briefing failed")
            return "Sorry, something went wrong while preparing your briefing."

    def _collect_memories(self):
        now = datetime.now().astimezone()
        start = now.date().isoformat()
        end = (now.date() + timedelta(days=LOOKAHEAD_DAYS)).isoformat()
        return self.store.get_by_date_range(start, end)

    def _summarize(self, memories) -> str:
        now = datetime.now().astimezone()
        data = [m.to_dict() for m in memories]
        return self.openai.summarize_briefing(data, str(now.date()), now.strftime("%A"))
