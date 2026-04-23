import logging
import asyncio
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings.config import settings

logger = logging.getLogger(__name__)


class ChunkingException(Exception):
    def __init__(self, file_name: str, section_index: int | str, reason: str):
        super().__init__(
            f"[Chunking] Failed on '{file_name}' section {section_index}: {reason}"
        )


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunking.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunking.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._domain_splitters: dict[str, RecursiveCharacterTextSplitter] = {}
        if not chunk_size and not chunk_overlap:
            for domain in (settings.chunking.domain_overrides or {}):
                cfg = settings.chunking.for_domain(domain)
                self._domain_splitters[domain] = RecursiveCharacterTextSplitter(
                    chunk_size=cfg.chunk_size,
                    chunk_overlap=cfg.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )

    def _get_splitter(self, domain: str) -> RecursiveCharacterTextSplitter:
        return self._domain_splitters.get(domain, self._default_splitter)

    async def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into flat chunks, carrying over source metadata."""
        if not docs:
            return []

        all_chunks: List[Document] = []
        tasks = [self._chunk_single(doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for doc, res in zip(docs, results):
            if isinstance(res, Exception):
                logger.error(
                    f"Failed to chunk '{doc.metadata.get('file_name', 'unknown')}' "
                    f"section {doc.metadata.get('section_index', '?')}: {res}"
                )
            else:
                all_chunks.extend(res)  # type: ignore[arg-type]

        logger.info(f"Chunked {len(docs)} sections → {len(all_chunks)} chunks")
        return all_chunks

    @staticmethod
    def _extract_heading(text: str) -> str:
        for line in text.splitlines():
            if line.startswith("#"):
                return line.strip()
        return ""

    async def _chunk_single(self, doc: Document) -> List[Document]:
        try:
            heading = self._extract_heading(doc.page_content)
            domain = doc.metadata.get("domain", "")
            splitter = self._get_splitter(domain)
            splits = await asyncio.to_thread(splitter.split_text, doc.page_content)
            chunks = []
            for i, text in enumerate(splits):
                text = text.strip()
                if not text:
                    continue
                # prepend heading to continuation chunks so context is not lost
                if i > 0 and heading and not text.startswith("#"):
                    text = f"{heading}\n{text}"
                chunks.append(Document(
                    page_content=text,
                    metadata={**doc.metadata, "chunk_index": i},
                ))
            return chunks
        except Exception as e:
            raise ChunkingException(
                file_name=doc.metadata.get("file_name", "unknown"),
                section_index=doc.metadata.get("section_index", "?"),
                reason=str(e),
            ) from e
