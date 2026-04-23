import uuid
import logging
import hashlib

import psycopg
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel, Field, computed_field
from psycopg.types.json import Json

from settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DocumentInsert(BaseModel):
    content: str
    embedding: list[float]
    source_file: str | None = None
    page_number: int | None = None
    chunk_index: int
    total_chunks: int
    access_level: str = "internal"
    created_by: str
    doc_type: str = "unknown"
    chunk_type: str = "text"
    extraction_method: str = "direct_read"
    extraction_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_length: int
    metadata: dict = Field(default_factory=dict)

    @computed_field
    @property
    def doc_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseVectorStore:
    async def insert(self, docs) -> None: ...
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list: ...
    async def add_documents(self, docs: list, embeddings: list[list[float]]) -> None:
        await self.insert(docs)


# ---------------------------------------------------------------------------
# PGVector
# ---------------------------------------------------------------------------

class PGVectorStore(BaseVectorStore):

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        conn = await psycopg.AsyncConnection.connect(settings.vector_store.connection_string)
        await register_vector_async(conn)
        return conn

    async def insert(self, docs) -> None:
        from rag.embeddings import EmbeddingService
        embed_model = EmbeddingService()
        embeddings = await embed_model.embed_documents(docs=docs)

        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                for i, (doc, emb) in enumerate(zip(docs, embeddings)):
                    row = DocumentInsert(
                        content=doc.page_content,
                        embedding=emb,
                        source_file=doc.metadata.get("source"),
                        page_number=doc.metadata.get("page"),
                        chunk_index=doc.metadata.get("chunk_index", i),
                        total_chunks=len(docs),
                        created_by="cortex",
                        doc_type=doc.metadata.get("file_type", "unknown"),
                        chunk_length=len(doc.page_content),
                        metadata=doc.metadata or {},
                    )
                    await cur.execute(
                        """
                        INSERT INTO documents (
                            content, embedding,
                            source_file, page_number, chunk_index, total_chunks, doc_hash,
                            access_level, created_by,
                            doc_type, chunk_type, extraction_method, extraction_confidence,
                            chunk_length, metadata
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            row.content, row.embedding, row.source_file, row.page_number,
                            row.chunk_index, row.total_chunks, row.doc_hash, row.access_level,
                            row.created_by, row.doc_type, row.chunk_type, row.extraction_method,
                            row.extraction_confidence, row.chunk_length, Json(row.metadata),
                        ),
                    )
            await conn.commit()

    async def add_documents(self, docs: list, embeddings: list[list[float]]) -> None:
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                for i, (doc, emb) in enumerate(zip(docs, embeddings)):
                    row = DocumentInsert(
                        content=doc.page_content,
                        embedding=emb,
                        source_file=doc.metadata.get("source"),
                        page_number=doc.metadata.get("page"),
                        chunk_index=doc.metadata.get("chunk_index", i),
                        total_chunks=len(docs),
                        created_by="cortex",
                        doc_type=doc.metadata.get("file_type", "unknown"),
                        chunk_length=len(doc.page_content),
                        metadata=doc.metadata or {},
                    )
                    await cur.execute(
                        """
                        INSERT INTO documents (
                            content, embedding,
                            source_file, page_number, chunk_index, total_chunks, doc_hash,
                            access_level, created_by,
                            doc_type, chunk_type, extraction_method, extraction_confidence,
                            chunk_length, metadata
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            row.content, row.embedding, row.source_file, row.page_number,
                            row.chunk_index, row.total_chunks, row.doc_hash, row.access_level,
                            row.created_by, row.doc_type, row.chunk_type, row.extraction_method,
                            row.extraction_confidence, row.chunk_length, Json(row.metadata),
                        ),
                    )
            await conn.commit()

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT content, source_file, page_number, metadata,
                           embedding <-> %s::vector AS distance
                    FROM documents ORDER BY distance LIMIT %s
                    """,
                    (query_embedding, top_k),
                )
                return await cur.fetchall()


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantVectorStore(BaseVectorStore):

    def _client(self):
        from qdrant_client import AsyncQdrantClient
        return AsyncQdrantClient(url=settings.vector_store.url, api_key=settings.vector_store.api_key)

    async def insert(self, docs) -> None:
        from qdrant_client.models import PointStruct
        from rag.embeddings import EmbeddingService
        embed_model = EmbeddingService()
        embeddings = await embed_model.embed_documents(docs=docs)

        points = []
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            row = DocumentInsert(
                content=doc.page_content, embedding=emb,
                source_file=doc.metadata.get("source"), page_number=doc.metadata.get("page"),
                chunk_index=i, total_chunks=len(docs), created_by="cortex",
                doc_type=doc.metadata.get("file_type", "unknown"),
                chunk_length=len(doc.page_content), metadata=doc.metadata or {},
            )
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, row.doc_hash)),
                vector=row.embedding,
                payload=row.model_dump(exclude={"embedding"}),
            ))

        async with self._client() as client:
            await client.upsert(collection_name=settings.vector_store.collection_name, points=points)

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        async with self._client() as client:
            return await client.search(
                collection_name=settings.vector_store.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )


# ---------------------------------------------------------------------------
# VectorStoreService — provider switch
# ---------------------------------------------------------------------------

class VectorStoreService:
    _instance: "VectorStoreService | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._backend = cls._instance._load_backend()
        return cls._instance

    def _load_backend(self) -> BaseVectorStore:
        provider = settings.vector_store.provider.lower()
        if provider == "pgvector":
            return PGVectorStore()
        if provider == "qdrant":
            return QdrantVectorStore()
        raise ValueError(f"Unsupported vector store provider: '{provider}'.")

    async def insert(self, docs) -> None:
        await self._backend.insert(docs)

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        return await self._backend.search(query_embedding, top_k)

    async def add_documents(self, docs: list, embeddings: list[list[float]]) -> None:
        await self._backend.add_documents(docs, embeddings)
