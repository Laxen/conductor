import json
import logging
import math
import os
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass

from integrations.openai import OpenAIIntegration


DB_PATH = "/data/memory.db"
logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    raw_text    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    due_date    TEXT,
    location    TEXT,
    tag         TEXT,
    embedding   TEXT NOT NULL
)
"""


_METADATA_FIELDS = ("due_date", "location", "tag")


@dataclass(frozen=True)
class Metadata:
    due_date: str | None = None
    location: str | None = None
    tag: str | None = None

    @classmethod
    def from_intent(cls, intent_response: dict) -> "Metadata":
        return cls(**{
            f: v if isinstance(v := intent_response.get(f), str) else None
            for f in _METADATA_FIELDS
        })

    def display(self) -> str:
        parts = [f"{f}: {getattr(self, f)}" for f in _METADATA_FIELDS if getattr(self, f)]
        return f" ({', '.join(parts)})" if parts else ""

    def has_any(self) -> bool:
        return any(getattr(self, f) is not None for f in _METADATA_FIELDS)

    def as_tuple(self) -> tuple[str | None, str | None, str | None]:
        return (self.due_date, self.location, self.tag)


@dataclass(frozen=True)
class Memory:
    id: str
    raw_text: str
    due_date: str | None
    location: str | None
    tag: str | None

    @property
    def metadata(self) -> Metadata:
        return Metadata(due_date=self.due_date, location=self.location, tag=self.tag)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "due_date": self.due_date,
            "location": self.location,
            "tag": self.tag,
        }


class MemoryStore:
    """Encapsulates all database operations for memories."""

    def __init__(self, db_path: str = DB_PATH):
        if not os.path.exists(os.path.dirname(db_path) or "."):
            db_path = "memory.db"
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def init(self) -> None:
        with self._conn() as c:
            c.execute(_CREATE_SQL)
            try:
                c.execute("ALTER TABLE memories ADD COLUMN tag TEXT")
            except sqlite3.OperationalError:
                pass

    def insert(self, raw_text: str, metadata: Metadata, embedding: list[float]) -> None:
        short_id = uuid.uuid4().hex[:8]
        with self._conn() as c:
            c.execute(
                "INSERT INTO memories (id, raw_text, created_at, due_date, location, tag, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    short_id,
                    raw_text,
                    datetime.now(timezone.utc).isoformat(),
                    *metadata.as_tuple(),
                    json.dumps(embedding),
                ),
            )

    def _fetch_all_rows(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, raw_text, due_date, location, tag, embedding FROM memories").fetchall()
        return [dict(r) for r in rows]

    def update(self, memory_id: str, raw_text: str, metadata: Metadata, embedding: list[float]) -> bool:
        with self._conn() as c:
            cursor = c.execute(
                "UPDATE memories SET raw_text = ?, due_date = ?, location = ?, tag = ?, embedding = ? WHERE id = ?",
                (raw_text, *metadata.as_tuple(), json.dumps(embedding), memory_id),
            )
        return cursor.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        with self._conn() as c:
            cursor = c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def get_all(self) -> list[Memory]:
        rows = self._fetch_all_rows()
        return [Memory(id=r["id"], raw_text=r["raw_text"], due_date=r["due_date"], location=r["location"], tag=r["tag"]) for r in rows]

    def retrieve_by_metadata(self, metadata: Metadata) -> list[Memory]:
        conditions: list[str] = []
        params: list[str] = []
        for field in _METADATA_FIELDS:
            value = getattr(metadata, field)
            if value:
                conditions.append(f"{field} = ?")
                params.append(value)

        if not conditions:
            return []

        where = " OR ".join(conditions)
        query = f"SELECT id, raw_text, due_date, location, tag FROM memories WHERE {where} ORDER BY created_at DESC"

        with self._conn() as c:
            rows = c.execute(query, tuple(params)).fetchall()

        return [Memory(id=r["id"], raw_text=r["raw_text"], due_date=r["due_date"], location=r["location"], tag=r["tag"]) for r in rows]

    def get_by_date_range(self, start_date: str, end_date: str) -> list[Memory]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, raw_text, due_date, location, tag FROM memories WHERE due_date >= ? AND due_date <= ? ORDER BY due_date ASC",
                (start_date, end_date),
            ).fetchall()
        return [Memory(id=r["id"], raw_text=r["raw_text"], due_date=r["due_date"], location=r["location"], tag=r["tag"]) for r in rows]

    def get_unique_tags(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT tag FROM memories WHERE tag IS NOT NULL").fetchall()
        return [r["tag"] for r in rows]

    def get_overdue(self, today: str) -> list[Memory]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, raw_text, due_date, location, tag FROM memories WHERE due_date < ? ORDER BY due_date ASC",
                (today,),
            ).fetchall()
        return [Memory(id=r["id"], raw_text=r["raw_text"], due_date=r["due_date"], location=r["location"], tag=r["tag"]) for r in rows]

    def delete_by_tag(self, tag: str) -> int:
        with self._conn() as c:
            cursor = c.execute("DELETE FROM memories WHERE tag = ?", (tag,))
        return cursor.rowcount

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def top_k(self, query_embedding: list[float], k: int = 5) -> list[Memory]:
        rows = self._fetch_all_rows()
        scored = [
            (
                self._cosine(query_embedding, json.loads(r["embedding"])),
                r["id"],
                r["raw_text"],
                r["due_date"],
                r["location"],
                r["tag"],
            )
            for r in rows
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        logger.info("Top %s scored memories:", k)
        for score, memory_id, text, due_date, location, tag in scored[:k]:
            logger.info("Score: %s, ID: %s, Memory: %s, Due Date: %s, Location: %s, Tag: %s", score, memory_id, text, due_date, location, tag)

        return [Memory(id=memory_id, raw_text=text, due_date=due_date, location=location, tag=tag) for score, memory_id, text, due_date, location, tag in scored[:k]]


class MemoryApp:
    """Handles memory management."""

    def __init__(self, store: MemoryStore, openai: OpenAIIntegration):
        self.store = store
        self.openai = openai

    def handle_input(self, text: str, username: str) -> str | None:
        logger.info("New input received")

        try:
            query_embedding = self.openai.embed(text)
            top_memories = self.store.top_k(query_embedding)
            available_tags = self.store.get_unique_tags()
            intent_responses = self.openai.extract_intent(text, top_memories, available_tags)
        except Exception:
            logger.exception("Failed to process incoming message")
            return "Sorry, something went wrong while processing your message."

        responses: list[str] = []
        for intent_response in intent_responses:
            intent = intent_response.get("intent", "unknown")
            metadata = Metadata.from_intent(intent_response)
            normalized_text = intent_response.get("normalized_text")
            current_top_memories = self.store.top_k(query_embedding)

            if intent == "store":
                responses.append(self._handle_store(text, query_embedding, normalized_text, metadata))
                continue

            if intent == "retrieve":
                if metadata.has_any():
                    memories = self.store.retrieve_by_metadata(metadata)
                    responses.append(self._handle_retrieve(text, memories))
                else:
                    responses.append(self._handle_retrieve(text, current_top_memories))
                continue

            if intent == "update":
                responses.append(self._handle_update(text, query_embedding, normalized_text, metadata, current_top_memories, intent_response))
                continue

            if intent == "delete":
                responses.append(self._handle_delete(current_top_memories, intent_response, metadata))
                continue

            responses.append("I couldn't determine what you'd like to do. Please try rephrasing.")

        if not responses:
            return "I couldn't determine what you'd like to do. Please try rephrasing."

        return "\n\n".join(responses)

    def handle_showdb(self) -> str:
        all_memories = self.store.get_all()
        if not all_memories:
            return "The memory DB is empty."

        lines = [f"[{m.id}] {m.raw_text}{m.metadata.display()}" for m in all_memories]
        return "Memory DB:\n" + "\n".join(lines)

    def handle_schedule(self) -> str:
        today = date.today()
        today_str = today.isoformat()

        sections: list[str] = []

        overdue = self.store.get_overdue(today_str)
        if overdue:
            lines = [f"• {m.raw_text}{m.metadata.display()}" for m in overdue]
            sections.append("Overdue\n" + "\n".join(lines))

        day_labels = ["Today", "Tomorrow"] + [
            (today + timedelta(days=i)).strftime("%A") for i in range(2, 8)
        ]
        for i, label in enumerate(day_labels):
            day_str = (today + timedelta(days=i)).isoformat()
            entries = self.store.get_by_date_range(day_str, day_str)
            if entries:
                lines = [f"• {m.raw_text}{m.metadata.display()}" for m in entries]
                sections.append(f"{label}\n" + "\n".join(lines))

        if not sections:
            return "No entries scheduled."

        return "\n\n".join(sections)

    def _resolve_target(self, top_memories: list[Memory], intent_response: dict) -> Memory:
        if not top_memories:
            raise ValueError("No memories available to target")

        candidate = intent_response.get("target_id")
        if isinstance(candidate, str) and candidate:
            for item in top_memories:
                if item.id == candidate:
                    return item
            raise ValueError(f"Target ID {candidate} not found in top memories")
        raise ValueError("No target_id specified")

    def _prepare_content_and_embedding(
        self,
        original_text: str,
        query_embedding: list[float],
        normalized_text: object,
    ) -> tuple[str, list[float]]:
        if isinstance(normalized_text, str) and normalized_text.strip():
            content = normalized_text
        else:
            content = original_text

        embedding = query_embedding if content == original_text else self.openai.embed(content)
        return content, embedding

    def _handle_store(self, text: str, query_embedding: list[float], normalized_text: object, metadata: Metadata) -> str:
        try:
            content, embedding = self._prepare_content_and_embedding(text, query_embedding, normalized_text)
            self.store.insert(content, metadata, embedding)
            return f"Store\n\"{content}\"{metadata.display()}."
        except Exception:
            logger.exception("Store failed")
            return "Sorry, I couldn't store that right now."

    def _handle_retrieve(self, text: str, top_memories: list[Memory]) -> str:
        try:
            if not top_memories:
                return "I don't have anything stored yet."

            contexts = [m.to_dict() for m in top_memories]
            return self.openai.answer_with_context(text, contexts)
        except Exception:
            logger.exception("Retrieve failed")
            return "Sorry, I couldn't retrieve that right now."

    def _handle_update(
        self, text: str, query_embedding: list[float], normalized_text: object,
        metadata: Metadata, top_memories: list[Memory], intent_response: dict,
    ) -> str:
        try:
            target = self._resolve_target(top_memories, intent_response)
            content, embedding = self._prepare_content_and_embedding(text, query_embedding, normalized_text)
            updated = self.store.update(target.id, content, metadata, embedding)
            if not updated:
                return "I couldn't find that memory to update."

            return f"Update\n\"{target.raw_text}\"{target.metadata.display()}\nto\n\"{content}\"{metadata.display()}."
        except ValueError as error:
            logger.error("Update target resolution failed: %s", error)
            return f"I couldn't update: {error}"
        except Exception:
            logger.exception("Update failed")
            return "Sorry, I couldn't update that right now."

    def _handle_delete(self, top_memories: list[Memory], intent_response: dict, metadata: Metadata) -> str:
        try:
            if metadata.tag and not intent_response.get("target_id"):
                count = self.store.delete_by_tag(metadata.tag)
                if count == 0:
                    return f"No memories found with tag \"{metadata.tag}\"."
                return f"Deleted {count} memory/memories with tag \"{metadata.tag}\"."

            target = self._resolve_target(top_memories, intent_response)
            deleted = self.store.delete(target.id)
            if not deleted:
                return "I couldn't find that memory to delete."

            return f"Delete\n\"{target.raw_text}\"{target.metadata.display()}."
        except ValueError as error:
            logger.error("Delete target resolution failed: %s", error)
            return f"I couldn't delete: {error}"
        except Exception:
            logger.exception("Delete failed")
            return "Sorry, I couldn't delete that right now."
