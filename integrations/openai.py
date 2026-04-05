import json
import logging
from datetime import datetime

from openai import OpenAI
from settings import get_env


logger = logging.getLogger(__name__)

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
                        "text": {"type": "string", "description": "Free text query for semantic similarity search (optional)."},
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
                    "The text field is used to find the entry via semantic search. "
                    "new_text is the updated content; provide all metadata fields in their final state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to search for the entry to update (semantic search)."},
                        "new_text": {"type": "string", "description": "The new text content for the entry."},
                        "due_date": {"type": "string", "description": "New due date in YYYY-MM-DD format (optional)."},
                        "location": {"type": "string", "description": "New location (optional)."},
                        "tag": {"type": "string", "description": "New tag/category (optional)."},
                    },
                    "required": ["text", "new_text"],
                },
            },
            {
                "type": "function",
                "name": "delete_entry",
                "description": (
                    "Delete an entry from the memory store. "
                    "Provide text for semantic search to delete a single matching entry, "
                    "or tag to delete all entries with that tag."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to find the entry to delete via semantic search (optional)."},
                        "tag": {"type": "string", "description": "Delete all entries with this tag (optional)."},
                    },
                },
            },
        ]

    def run_tool_call(self, text: str, available_tags: list[str] | None = None) -> object:
        """Turn 1: call the LLM with tools and return the raw response."""
        now = datetime.now().astimezone()
        tools = self._build_tools(available_tags)
        instructions = (
            "You are an assistant managing a memory-base for a user. "
            "Use the available tools to fulfil the user's request. "
            f"Reference date: {now.date()} ({now.strftime('%A')})."
        )

        logger.info("[LLM][tool_call][request] model=%s\ninput=%s", self.intent_model, text)

        response = self.client.responses.create(
            model=self.intent_model,
            instructions=instructions,
            input=[{"role": "user", "content": text}],
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

    def answer_with_tool_results(
        self,
        original_query: str,
        get_entries_calls: list,
        tool_results: list[tuple[str, str]],
    ) -> str:
        """Turn 2: answer the user's query using get_entries results. No tools are provided."""
        now = datetime.now().astimezone()
        instructions = (
            "You are an assistant managing a memory-base for a user. "
            "Use ONLY the provided data to answer the user's original question. "
            f"Reference date: {now.date()} ({now.strftime('%A')})."
        )

        input_items: list = [
            {"role": "user", "content": original_query},
            *get_entries_calls,
            *[
                {"type": "function_call_output", "call_id": call_id, "output": output}
                for call_id, output in tool_results
            ],
        ]

        logger.info("[LLM][answer][request] model=%s\ninput=%s", self.intent_model, original_query)

        response = self.client.responses.create(
            model=self.intent_model,
            instructions=instructions,
            input=input_items,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        logger.info(
            "[LLM][answer][response] model=%s input_tokens=%s output_tokens=%s\noutput=%s",
            self.intent_model,
            input_tokens,
            output_tokens,
            response.output_text,
        )

        return response.output_text

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