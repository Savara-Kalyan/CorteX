from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_MSG_TYPES: dict[str, type] = {
    "HumanMessage": HumanMessage,
    "AIMessage": AIMessage,
    "SystemMessage": SystemMessage,
}

MAX_TURNS = 6
SESSION_TTL = 3_600


class RedisSessionManager:
    PREFIX = "session:"

    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self._redis = aioredis.Redis(
            host=host, port=port, db=db, decode_responses=True,
            socket_connect_timeout=2,
        )
        self._store: dict[str, list[dict]] = {}

    async def save_session(self, session_id: str, messages: list[Any]) -> None:
        serialized = [
            {"type": m.__class__.__name__, "content": m.content} for m in messages
        ]
        payload = json.dumps(serialized)
        try:
            await self._redis.setex(self._key(session_id), SESSION_TTL, payload)
        except Exception:
            self._store[session_id] = serialized
        logger.debug("session %s saved (%d messages)", session_id, len(messages))

    async def load_session(self, session_id: str) -> list[Any]:
        try:
            raw = await self._redis.get(self._key(session_id))
            if raw is None:
                return []
            serialized = json.loads(raw)
        except Exception:
            serialized = self._store.get(session_id, [])
        messages = [
            _MSG_TYPES[m["type"]](content=m["content"])
            for m in serialized
            if m["type"] in _MSG_TYPES
        ]
        logger.debug("session %s loaded (%d messages)", session_id, len(messages))
        return messages

    async def append_turn(
        self,
        session_id: str,
        human_text: str,
        ai_text: str,
    ) -> list[Any]:
        history = await self.load_session(session_id)
        history.append(HumanMessage(content=human_text))
        history.append(AIMessage(content=ai_text))
        max_msgs = MAX_TURNS * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        await self.save_session(session_id, history)
        return history

    async def delete_session(self, session_id: str) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except Exception:
            self._store.pop(session_id, None)

    async def extend_ttl(self, session_id: str) -> None:
        try:
            await self._redis.expire(self._key(session_id), SESSION_TTL)
        except Exception:
            pass

    def _key(self, session_id: str) -> str:
        return f"{self.PREFIX}{session_id}"
