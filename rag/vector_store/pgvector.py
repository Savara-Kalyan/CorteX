import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector_async

import logging

from settings import settings
from rag.embeddings.service import EmbeddingService
from rag.vector_store.base import BaseVectorStore
from rag.vector_store.model import DocumentInsert

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):

    embed_model: EmbeddingService = EmbeddingService()

    async def insert(self, docs) -> None:
        logger.info("Inserting documents into PGVector: doc_count=%s", len(docs))

        try:
            embeddings = await self.embed_model.embed_documents(docs=docs)
        except Exception as e:
            logger.error("Embedding generation failed: error=%s doc_count=%s", e, len(docs))
            raise RuntimeError("Failed to generate embeddings before PGVector insert") from e

        try:
            async with await psycopg.AsyncConnection.connect(
                settings.vector_store.connection_string
            ) as conn:
                await register_vector_async(conn)

                async with conn.cursor() as cur:
                    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
                        row = DocumentInsert(
                            content=doc.page_content,
                            embedding=emb,
                            source_file=doc.metadata.get("source"),
                            page_number=doc.metadata.get("page"),
                            chunk_index=i,
                            total_chunks=len(docs),
                            created_by="kalyan",
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
                                doc_type, chunk_type, extraction_method, extraction_confidence, chunk_length,
                                metadata
                            )
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                row.content,
                                row.embedding,
                                row.source_file,
                                row.page_number,
                                row.chunk_index,
                                row.total_chunks,
                                row.doc_hash,
                                row.access_level,
                                row.created_by,
                                row.doc_type,
                                row.chunk_type,
                                row.extraction_method,
                                row.extraction_confidence,
                                row.chunk_length,
                                Json(row.metadata),
                            ),
                        )

                    await conn.commit()

        except psycopg.OperationalError as e:
            logger.error("PGVector connection failed: error=%s", e)
            raise RuntimeError("Could not connect to PostgreSQL") from e
        except psycopg.DatabaseError as e:
            logger.error("PGVector insert query failed: error=%s", e)
            raise RuntimeError("Failed to insert documents into PostgreSQL") from e

        logger.info("PGVector insert complete: doc_count=%s", len(docs))

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        logger.info("Searching PGVector: top_k=%s", top_k)

        try:
            async with await psycopg.AsyncConnection.connect(
                settings.vector_store.connection_string
            ) as conn:
                await register_vector_async(conn)

                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT content, source_file, page_number, metadata,
                               embedding <-> %s::vector AS distance
                        FROM documents
                        ORDER BY distance
                        LIMIT %s
                        """,
                        (query_embedding, top_k),
                    )
                    results = await cur.fetchall()

        except psycopg.OperationalError as e:
            logger.error("PGVector connection failed during search: error=%s", e)
            raise RuntimeError("Could not connect to PostgreSQL") from e
        except psycopg.DatabaseError as e:
            logger.error("PGVector search query failed: error=%s", e)
            raise RuntimeError("Failed to search documents in PostgreSQL") from e

        logger.info("PGVector search complete: result_count=%s", len(results))
        return results
