from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_memory (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    confidence  FLOAT DEFAULT 1.0,
    source      TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW(),
    expires_at  TIMESTAMP,
    UNIQUE(user_id, key)
);
CREATE INDEX IF NOT EXISTS idx_user_memory_user_id ON user_memory(user_id);
"""

_UPSERT_SQL = """
INSERT INTO user_memory (user_id, memory_type, key, value, source)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (user_id, key)
DO UPDATE SET
    value       = EXCLUDED.value,
    memory_type = EXCLUDED.memory_type,
    source      = EXCLUDED.source,
    updated_at  = NOW();
"""

_SELECT_SQL = "SELECT key, value, memory_type, source FROM user_memory WHERE user_id = %s"
_SELECT_TYPED_SQL = _SELECT_SQL + " AND memory_type = %s"


class LongTermMemoryStore:
    def __init__(self):
        self._db: dict[str, dict[str, dict]] = {}
        self._conn = None
        self._dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")

    async def _ensure_connection(self):
        if self._conn is not None:
            return
        if not self._dsn:
            return
        try:
            import psycopg
            self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
            async with self._conn.cursor() as cur:
                await cur.execute(_SCHEMA_SQL)
            logger.info("LongTermMemoryStore connected to PostgreSQL")
        except Exception as exc:
            logger.warning("LongTermMemoryStore PostgreSQL unavailable (%s) — using in-memory", exc)
            self._conn = None

    async def store(
        self,
        user_id: str,
        key: str,
        value: str,
        memory_type: str = "fact",
        source: str = "user_stated",
    ) -> None:
        await self._ensure_connection()
        if self._conn:
            async with self._conn.cursor() as cur:
                await cur.execute(_UPSERT_SQL, (user_id, memory_type, key, value, source))
        else:
            self._db.setdefault(user_id, {})[key] = {
                "value": value,
                "type": memory_type,
                "source": source,
                "created_at": datetime.utcnow().isoformat(),
            }
        logger.debug("stored memory user=%s key=%s", user_id, key)

    async def get(self, user_id: str, memory_type: str | None = None) -> dict[str, dict]:
        await self._ensure_connection()
        if self._conn:
            async with self._conn.cursor() as cur:
                if memory_type:
                    await cur.execute(_SELECT_TYPED_SQL, (user_id, memory_type))
                else:
                    await cur.execute(_SELECT_SQL, (user_id,))
                rows = await cur.fetchall()
            return {row[0]: {"value": row[1], "type": row[2], "source": row[3]} for row in rows}
        user_mems = self._db.get(user_id, {})
        if memory_type:
            return {k: v for k, v in user_mems.items() if v.get("type") == memory_type}
        return dict(user_mems)

    async def format_for_prompt(self, user_id: str) -> str:
        memories = await self.get(user_id)
        if not memories:
            return ""
        lines = ["Known information about this user:"]
        for key, data in memories.items():
            lines.append(f"  - {key}: {data['value']} (source: {data['source']})")
        return "\n".join(lines)

    async def extract_and_store(self, user_id: str, conversation_text: str, llm: Any) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await llm.ainvoke([
            SystemMessage(content=(
                "Extract memorable facts about the user from this conversation.\n"
                'Return JSON: {"facts": [{"key": "fact_name", "value": "fact_value", '
                '"type": "preference|fact|complaint|history"}]}\n'
                "Only include facts worth remembering long-term. Skip small talk.\n"
                'If nothing memorable: {"facts": []}'
            )),
            HumanMessage(content=conversation_text),
        ])
        try:
            data = json.loads(response.content)
            for fact in data.get("facts", []):
                await self.store(
                    user_id,
                    fact["key"],
                    fact["value"],
                    memory_type=fact.get("type", "fact"),
                    source="inferred",
                )
        except Exception as exc:
            logger.warning("Failed to parse LLM memory extraction: %s", exc)
