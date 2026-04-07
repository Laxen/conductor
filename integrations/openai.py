import json
import logging
from datetime import datetime

from openai import OpenAI
from settings import get_env


logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "You are an assistant managing a memory-base for a user, with entries being tasks, events or general information. "
    "Use the available tools to fully understand the user's intention and fulfill their request. "
    "If the user states a fact/task/reminder/event, check if there's already a closely matching entry that can be updated with that info, otherwise add a new one. "
    "If the user says something is done/finished or checked off, get the entry they're talking about and delete it. "
    "NEVER add extra text to an entry that the user didn't input. NEVER add metadata (date, location, tag, etc.) unless it's obvious from the user's message. "
    "NEVER lie about what you've done or make things up, if you don't understand the user's request then tell them why. "
    "Any relative date phrases like 'tomorrow' and weekdays like 'on Thursday' should be calculated deterministically from the given reference date. NEVER guess. "
    "When you have finished all necessary actions, respond with a summary of what you've done."
)

class OpenAIIntegration:
    def __init__(self):
        api_key = get_env("OPENAI_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAI api_key must be a non-empty string")

        self.client = OpenAI(api_key=api_key)
        self.intent_model = get_env("OPENAI_CHAT_MODEL")
        self.embedding_model = get_env("OPENAI_EMBEDDING_MODEL")

    def send_prompt(self, prompt: str, model: str, step: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.info("[LLM][%s][request] model=%s\ninput=%s", step, model, prompt)

        response = self.client.responses.create(
            model=model,
            input=prompt,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None

        logger.info(
            "[LLM][%s][response] model=%s input_tokens=%s output_tokens=%s\noutput=%s",
            step,
            model,
            input_tokens,
            output_tokens,
            response.output_text,
        )

        return response.output_text

    def _build_tools(self, available_tags: list[str] | None = None) -> list[dict]:
        tag_desc = "Tag/category for the entry (optional)."
        if available_tags:
            tag_desc += f" Known tags: {', '.join(available_tags)}."

        return [
            {
                "type": "function",
                "name": "add_entry",
                "description": "Add a new entry to the memory store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text content of the entry."},
                        "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format (optional)."},
                        "location": {"type": "string", "description": "Location associated with the entry (optional)."},
                        "tag": {"type": "string", "description": tag_desc},
                    },
                    "required": ["text"],
                },
            },
            {
                "type": "function",
                "name": "get_entries",
                "description": (
                    "Retrieve entries from the memory store. "
                    "Use metadata filters (due_date range, location, tag) for structured queries, "
                    "or the text field for free-text semantic/vector search. "
                    "Multiple filters are combined with AND."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Free text query for semantic similarity search, use only if you have no other way of filtering data (optional)."},
                        "due_date_start": {"type": "string", "description": "Start of due date range in YYYY-MM-DD format (optional)."},
                        "due_date_end": {"type": "string", "description": "End of due date range in YYYY-MM-DD format (optional)."},
                        "location": {"type": "string", "description": "Filter by location (optional)."},
                        "tag": {"type": "string", "description": "Filter by tag/category (optional)."},
                    },
                },
            },
            {
                "type": "function",
                "name": "update_entry",
                "description": (
                    "Update an existing entry in the memory store. "
                    "Use get_entries first to retrieve the entry; the returned entries include an id field to use here. "
                    "new_text is the updated content; provide all metadata fields in their final state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The id of the entry to update, obtained from get_entries."},
                        "new_text": {"type": "string", "description": "The new text content for the entry."},
                        "due_date": {"type": "string", "description": "New due date in YYYY-MM-DD format (optional)."},
                        "location": {"type": "string", "description": "New location (optional)."},
                        "tag": {"type": "string", "description": "New tag/category (optional)."},
                    },
                    "required": ["id", "new_text"],
                },
            },
            {
                "type": "function",
                "name": "delete_entry",
                "description": (
                    "Delete entries from the memory store. "
                    "Use get_entries first to find the entry and obtain its id for single deletions. "
                    "Alternatively, provide a tag to bulk-delete all entries with that tag."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The id of the entry to delete, obtained from get_entries (optional)."},
                        "tag": {"type": "string", "description": "Delete all entries with this tag (optional)."},
                    },
                },
            },
        ]

    def call_with_tools(self, conversation: list, available_tags: list[str] | None = None, instructions: str | None = None) -> object:
        """Call the LLM with tools and the full conversation history. Returns the raw response."""
        now = datetime.now().astimezone()
        tools = self._build_tools(available_tags)
        base = instructions if instructions is not None else DEFAULT_INSTRUCTIONS
        full_instructions = base + f"\nReference date: {now.date()} ({now.strftime('%A')})."

        logger.info("[LLM][tool_call][request] model=%s\nconversation_length=%s", self.intent_model, len(conversation))

        response = self.client.responses.create(
            model=self.intent_model,
            instructions=full_instructions,
            input=conversation,
            tools=tools,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        logger.info(
            "[LLM][tool_call][response] model=%s input_tokens=%s output_tokens=%s\noutput=%s",
            self.intent_model,
            input_tokens,
            output_tokens,
            response.output,
        )

        return response

    def embed(self, text: str) -> list[float]:
        logger.info("[LLM][embed][request] model=%s\ninput=%s", self.embedding_model, text)
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        embedding = response.data[0].embedding
        logger.info(
            "[LLM][embed][response] model=%s tokens_in=%s",
            self.embedding_model,
            prompt_tokens,
        )

        return embedding

    def summarize_briefing(self, memories: list[dict], reference_date: str, reference_day: str) -> str:
        data = json.dumps(memories)

        prompt = (
            "You are a personal briefing assistant. Given the user's scheduled tasks and events, produce a concise daily briefing. "
            "Start with today's scheduled items clearly listed. "
            "Then give a brief lookahead of items for the next 2 days. "
            "If there are no items for a particular day, say so. "
            "Keep it short and actionable. Do not invent items that are not in the data. "
            "\n\n"
            f"Today: {reference_date} ({reference_day})\n"
            f"Scheduled items: {data}"
        )

        return self.send_prompt(prompt, model=self.intent_model, step="briefing")
