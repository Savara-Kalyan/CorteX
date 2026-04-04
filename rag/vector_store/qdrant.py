import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

import logging

from settings import settings
from rag.embeddings.service import EmbeddingService
from rag.vector_store.base import BaseVectorStore
from rag.vector_store.model import DocumentInsert

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):

    embed_model: EmbeddingService = EmbeddingService()

    def _client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=settings.vector_store.url,
            api_key=settings.vector_store.api_key,
        )

    async def insert(self, docs) -> None:
        logger.info("Inserting documents into Qdrant: doc_count=%s", len(docs))

        try:
            embeddings = await self.embed_model.embed_documents(docs=docs)
        except Exception as e:
            logger.error("Embedding generation failed: error=%s doc_count=%s", e, len(docs))
            raise RuntimeError("Failed to generate embeddings before Qdrant insert") from e

        points = []
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

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, row.doc_hash)),
                    vector=row.embedding,
                    payload=row.model_dump(exclude={"embedding"}),
                )
            )

        try:
            async with self._client() as client:
                await client.upsert(
                    collection_name=settings.vector_store.collection_name,
                    points=points,
                )
        except UnexpectedResponse as e:
            logger.error("Qdrant upsert unexpected response: error=%s collection=%s", e, settings.vector_store.collection_name)
            raise RuntimeError("Qdrant returned an unexpected response during insert") from e
        except Exception as e:
            logger.error("Qdrant insert failed: error=%s collection=%s", e, settings.vector_store.collection_name)
            raise RuntimeError("Failed to insert documents into Qdrant") from e

        logger.info("Qdrant insert complete: doc_count=%s", len(docs))

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list:
        logger.info("Searching Qdrant: top_k=%s", top_k)

        try:
            async with self._client() as client:
                results = await client.search(
                    collection_name=settings.vector_store.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                )
        except UnexpectedResponse as e:
            logger.error("Qdrant search unexpected response: error=%s collection=%s", e, settings.vector_store.collection_name)
            raise RuntimeError("Qdrant returned an unexpected response during search") from e
        except Exception as e:
            logger.error("Qdrant search failed: error=%s collection=%s", e, settings.vector_store.collection_name)
            raise RuntimeError("Failed to search documents in Qdrant") from e

        logger.info("Qdrant search complete: result_count=%s", len(results))
        return results
