import json
import logging
import math
import os
import sqlite3
import uuid
from collections.abc import Callable
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
_DATE_MIN = "0000-01-01"
_DATE_MAX = "9999-12-31"
_MAX_AGENTIC_LOOPS = 6


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

    def get_by_value(self, value: str) -> list[Memory]:
        """Return memories where any metadata field equals the given value."""
        conditions = [f"{field} = ?" for field in _METADATA_FIELDS]
        where = " OR ".join(conditions)
        params = tuple(value for _ in _METADATA_FIELDS)
        query = f"SELECT id, raw_text, due_date, location, tag FROM memories WHERE {where} ORDER BY created_at DESC"
        with self._conn() as c:
            rows = c.execute(query, params).fetchall()
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

    def handle_input(self, text: str, username: str, confirm_fn: Callable[[str, dict], bool] | None = None) -> str | None:
        logger.info("New input received")

        try:
            available_tags = self.store.get_unique_tags()
        except Exception:
            logger.exception("Failed to fetch available tags")
            return "Sorry, something went wrong while processing your message."

        conversation: list = [{"role": "user", "content": text}]
        last_response = None

        for loop_idx in range(_MAX_AGENTIC_LOOPS):
            try:
                last_response = self.openai.call_with_tools(conversation, available_tags)
            except Exception:
                logger.exception("Failed to call LLM (loop %s)", loop_idx + 1)
                return "Sorry, something went wrong while processing your message."

            conversation.extend(last_response.output)

            tool_calls = [item for item in last_response.output if getattr(item, "type", None) == "function_call"]

            if not tool_calls:
                return last_response.output_text or "Done."

            for tc in tool_calls:
                args = json.loads(tc.arguments)
                logger.info("[tool_call] name=%s args=%s", tc.name, args)

                if tc.name in ("update_entry", "delete_entry") and confirm_fn is not None:
                    if not confirm_fn(tc.name, args):
                        return "Operation cancelled."

                result = self._execute_tool(tc.name, args)
                logger.info("[tool_call] result=%s", result)
                conversation.append({"type": "function_call_output", "call_id": tc.call_id, "output": result})

        logger.error("Agentic loop exhausted maximum %s iterations without the LLM finishing tool execution", _MAX_AGENTIC_LOOPS)
        return "Execution stopped due to max number of iterations reached."

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "add_entry":
            return self._tool_add_entry(args)
        if name == "get_entries":
            entries = self._tool_get_entries(args)
            return json.dumps([m.to_dict() for m in entries])
        if name == "update_entry":
            return self._tool_update_entry(args)
        if name == "delete_entry":
            return self._tool_delete_entry(args)
        logger.error("Unknown tool called by LLM: %s", name)
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _tool_add_entry(self, args: dict) -> str:
        try:
            text = args.get("text", "")
            metadata = Metadata(
                due_date=args.get("due_date"),
                location=args.get("location"),
                tag=args.get("tag"),
            )
            embedding = self.openai.embed(text)
            self.store.insert(text, metadata, embedding)
            return json.dumps({"status": "success", "text": text})
        except Exception:
            logger.exception("add_entry tool failed")
            return json.dumps({"error": "Failed to store entry."})

    def _tool_get_entries(self, args: dict) -> list[Memory]:
        try:
            text = args.get("text")
            due_date_start = args.get("due_date_start")
            due_date_end = args.get("due_date_end")
            location = args.get("location")
            tag = args.get("tag")

            result_ids: set[str] | None = None
            all_results: dict[str, Memory] = {}

            if due_date_start or due_date_end:
                start = due_date_start or _DATE_MIN
                end = due_date_end or _DATE_MAX
                memories = self.store.get_by_date_range(start, end)
                ids = {m.id for m in memories}
                result_ids = ids if result_ids is None else result_ids & ids
                all_results.update({m.id: m for m in memories})

            if location or tag:
                meta_filter = Metadata(location=location, tag=tag)
                memories = self.store.retrieve_by_metadata(meta_filter)
                ids = {m.id for m in memories}
                result_ids = ids if result_ids is None else result_ids & ids
                all_results.update({m.id: m for m in memories})

            if text:
                embedding = self.openai.embed(text)
                memories = self.store.top_k(embedding)
                ids = {m.id for m in memories}
                result_ids = ids if result_ids is None else result_ids & ids
                all_results.update({m.id: m for m in memories})

            if result_ids is None:
                return []

            return [all_results[mid] for mid in result_ids]
        except Exception:
            logger.exception("get_entries tool failed")
            return []

    def _tool_update_entry(self, args: dict) -> str:
        try:
            entry_id = args.get("id", "")
            new_text = args.get("new_text", "")
            metadata = Metadata(
                due_date=args.get("due_date"),
                location=args.get("location"),
                tag=args.get("tag"),
            )
            new_embedding = self.openai.embed(new_text)
            updated = self.store.update(entry_id, new_text, metadata, new_embedding)
            if not updated:
                return json.dumps({"error": f"No entry found with id \"{entry_id}\"."})
            return json.dumps({"status": "success", "id": entry_id, "new_text": new_text})
        except Exception:
            logger.exception("update_entry tool failed")
            return json.dumps({"error": "Failed to update entry."})

    def _tool_delete_entry(self, args: dict) -> str:
        try:
            entry_id = args.get("id")
            tag = args.get("tag")

            if tag and not entry_id:
                count = self.store.delete_by_tag(tag)
                if count == 0:
                    return json.dumps({"error": f"No entries found with tag \"{tag}\"."})
                return json.dumps({"status": "success", "deleted_count": count, "tag": tag})

            if entry_id:
                deleted = self.store.delete(entry_id)
                if not deleted:
                    return json.dumps({"error": f"No entry found with id \"{entry_id}\"."})
                return json.dumps({"status": "success", "id": entry_id})

            return json.dumps({"error": "Provide either id or tag to identify entries to delete."})
        except Exception:
            logger.exception("delete_entry tool failed")
            return json.dumps({"error": "Failed to delete entry."})

    def handle_show(self, args: list[str]) -> str:
        value = " ".join(args) if args else None

        if value is not None:
            memories = self.store.get_by_value(value)
        else:
            memories = self.store.get_all()

        if not memories:
            return f"No entries found for '{value}'." if value else "The memory DB is empty."

        lines = [f"• {m.raw_text}{m.metadata.display()}" for m in memories]
        return "\n".join(lines)

    def handle_schedule(self, args: list[str]) -> str:
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
