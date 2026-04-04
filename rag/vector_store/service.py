import logging

from settings import settings
from rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorStoreService:

    _instance: "VectorStoreService | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance._backend = cls._instance._load_backend()
            except Exception:
                cls._instance = None
                raise
        return cls._instance

    def _load_backend(self) -> BaseVectorStore:
        provider = settings.vector_store.provider.lower()
        logger.info("Loading vector store backend: provider=%s", provider)

        if provider == "pgvector":
            from rag.vector_store.pgvector import PGVectorStore
            return PGVectorStore()

        if provider == "qdrant":
            from rag.vector_store.qdrant import QdrantVectorStore
            return QdrantVectorStore()

        raise ValueError(
            f"Unsupported vector store provider: '{provider}'. Choose 'pgvector' or 'qdrant'."
        )

    async def insert(self, docs) -> None:
        try:
            await self._backend.insert(docs)
        except Exception as e:
            logger.error("Vector store insert failed: error=%s doc_count=%s", e, len(docs))
            raise

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        try:
            return await self._backend.search(query_embedding, top_k)
        except Exception as e:
            logger.error("Vector store search failed: error=%s top_k=%s", e, top_k)
            raise
