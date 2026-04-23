from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag.access_control import AccessControlService
from rag.chunking import DocumentChunker
from rag.embeddings import EmbeddingService
from rag.hybrid_search import HybridSearchRequest, HybridSearchService
from rag.ingestion.document_loader import DocumentIngestionService
from rag.query_understanding import QueryUnderstanding
from rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, llm: Any = None):
        self._ingestion = DocumentIngestionService()
        self._chunker = DocumentChunker()
        self._embeddings = EmbeddingService()
        self._vector_store = VectorStoreService()
        self._query_understanding = QueryUnderstanding()
        self._access_control = AccessControlService()
        self._hybrid_search = HybridSearchService()
        self._llm = llm or self._load_llm()


    async def ingest(self, directory: str | Path) -> dict:
        directory = Path(directory)
        logger.info("Ingesting documents from %s", directory)

        docs = self._ingestion.load_directory(str(directory))
        logger.info("L1 ingestion: %d documents loaded", len(docs))

        chunks = await self._chunker.chunk_documents(docs)
        logger.info("L2 chunking: %d chunks created", len(chunks))

        texts = [c.page_content for c in chunks]
        embeddings = await self._embeddings.embed_documents(texts)
        logger.info("L3 embeddings: %d vectors generated", len(embeddings))

        await self._vector_store.add_documents(chunks, embeddings)
        logger.info("L4 vector store: documents indexed")

        return {"documents_ingested": len(docs), "chunks_created": len(chunks)}


    async def query(
        self,
        query: str,
        user_id: str = "anonymous",
        user_tier: str = "public",
        top_k: int | None = None,
        system_prompt: str | None = None,
        domain: str | None = None,
    ) -> dict:
        logger.info("RAG query: user=%s tier=%s query=%r", user_id, user_tier, query[:80])

        qu_result = await self._query_understanding.process(query)
        reformulated = qu_result.get("reformulated", query)
        intent = qu_result.get("intent", "unknown")
        answerable = qu_result.get("answerable", True)

        if not answerable:
            return {
                "answer": "I couldn't find relevant information for that query in our knowledge base.",
                "sources": [],
                "intent": intent,
                "reformulated_query": reformulated,
                "chunks_retrieved": 0,
            }

        from settings import settings as _settings
        effective_top_k = top_k if top_k is not None else _settings.retrieval.top_k
        search_request = HybridSearchRequest(
            query=reformulated,
            top_k=effective_top_k,
            fetch_k=_settings.retrieval.fetch_k,
            user_tier=user_tier,
            domain=domain,
        )
        search_response = await self._hybrid_search.search(search_request)
        chunks = search_response.results

        chunks = await self._access_control.filter(chunks, user_tier=user_tier)

        logger.info("Retrieved %d chunks after access control", len(chunks))

        if not chunks:
            return {
                "answer": "No relevant documents found in the knowledge base for your query.",
                "sources": [],
                "intent": intent,
                "reformulated_query": reformulated,
                "chunks_retrieved": 0,
            }

        context = "\n\n".join(
            f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.content}"
            for c in chunks
        )
        answer = await self._synthesise(query, context, system_prompt=system_prompt)
        sources = list({c.metadata.get("source", "unknown") for c in chunks})
        contexts = [c.content for c in chunks]
        chunk_ids = [int(c.id) for c in chunks if c.id]

        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "chunk_ids": chunk_ids,
            "intent": intent,
            "reformulated_query": reformulated,
            "chunks_retrieved": len(chunks),
        }


    async def _synthesise(self, query: str, context: str, system_prompt: str | None = None) -> str:
        import asyncio

        system = system_prompt or (
            "You are the Cortex Knowledge Agent. Your only source of truth is the CONTEXT block below.\n"
            "Rules:\n"
            "1. Answer using ONLY information explicitly stated in the context. No outside knowledge.\n"
            "2. Be concise — one to three sentences maximum.\n"
            "3. If the context does not contain the answer, reply: 'The provided documents do not cover this.'\n"
            "4. Do not paraphrase beyond what is needed to form a grammatical sentence."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"),
        ]
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._llm.invoke(messages)
        )
        return response.content

    @staticmethod
    def _load_llm():
        from settings import settings
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm.model, temperature=0.2)
