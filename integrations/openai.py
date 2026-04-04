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

    def extract_intent(self, text: str, top_memories: list, available_tags: list[str] | None = None) -> list[dict]:
        now = datetime.now().astimezone()

        memories_json = json.dumps([m.to_dict() for m in top_memories])
        
        prompt = (
            "You are an assistant managing a memory-base for a user. Your task is to extract the user's intent from their message. "
            "Reply with newline-delimited JSON only and no extra text. "
            "Each line must be one valid JSON object with this shape: {\"intent\": \"intent\", \"due_date\": \"YYYY-MM-DD or null\", \"location\": \"string or null\", \"tag\": \"string or null\", \"target_id\": \"string or null\", \"normalized_text\": \"string or null\"}. "
            "Return one line per action in the user's message, in user order. "
            "If the user asks for multiple actions, return multiple lines. "
            "The intent can be one of the following: "
            "\"store\" when the user inputs a fact/todo/reminder/event, "
            "\"retrieve\" when the user is asking a question, "
            "\"update\" when the user wants to modify an existing stored memory, "
            "\"delete\" when the user wants to remove an existing stored memory, "
            "\"unknown\" otherwise. "
            "intent is always required on every line. All other fields are optional unless stated below. "
            "When the intent is \"update\", target_id is required and must be one of the IDs from the top memories. "
            "When the intent is \"delete\", either target_id (one of the IDs from the top memories) or tag is required. Use tag without target_id when the user wants to delete all memories with that tag. "
            "When the intent is \"update\", normalized_text must be the final canonical form of the memory, with transitional or conversational words removed. Don't include the date in the text. "
            "location should be set only when the message explicitly implies a location context, otherwise null. "
            "tag should be set when the user explicitly mentions or implies a category or tag for the memory, otherwise null. "
            "For retrieve and delete intents, tag must be one of the available tags if provided. For store intent, tag can be a new value. "
            # "When not applicable, location, target_id and normalized_text should be null. "
            "due_date is set only when the message implies a deadline or date, otherwise null. "
            "due_date should be calculated deterministically from the given reference date, interpreting relative date phrases like 'tomorrow' and weekdays like 'on Thursday' accordingly. Never guess. "
            "\n\n"
            f"Reference date: {now.date()} ({now.strftime('%A')})\n"
            f"Available tags: {json.dumps(available_tags or [])}\n"
            f"Top memories: {memories_json}\n"
            f"User message: {text}"
        )

        output = self.send_prompt(prompt, model=self.intent_model, step="intent")
        try:
            intents: list[dict] = []
            rows = [line.strip() for line in output.splitlines() if line.strip()]

            for row in rows:
                payload = json.loads(row)
                if not isinstance(payload, dict):
                    raise ValueError("Each output row must be a JSON object")

                intents.append(
                    {
                        "intent": payload.get("intent", "unknown"),
                        "due_date": payload.get("due_date"),
                        "location": payload.get("location"),
                        "tag": payload.get("tag"),
                        "target_id": payload.get("target_id"),
                        "normalized_text": payload.get("normalized_text"),
                    }
                )

            if not intents:
                return [{"intent": "unknown", "due_date": None, "location": None, "tag": None, "target_id": None, "normalized_text": None}]

            return intents
        except Exception:
            logger.exception("Intent extraction failed. Raw output:\n%s", output)
            raise

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

    def answer_with_context(self, query: str, contexts: list) -> str:
        ctx = json.dumps(contexts)
        now = datetime.now().astimezone()
        
        prompt = (
            "You are an assistant managing a memory-base for a user. Your task is to answer the user's message using ONLY the information provided below. "
            "Use all of the data if needed to give a complete answer. "
            "\n\n"
            f"Reference date: {now.date()} ({now.strftime('%A')})\n"
            f"Data: {ctx}\n"
            f"User message: {query}"
        )

        output = self.send_prompt(prompt, model=self.intent_model, step="reasoning")
        return output

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