import uuid
import logging
import asyncio
from typing import List, Tuple

from chonkie import Pipeline
from langchain_core.documents import Document

logger = logging.getLogger("EnterpriseIngestion")


class ChunkingException(Exception):
    def __init__(self, file_name: str, section_index: int | str, reason: str):
        super().__init__(
            f"[Chunking] Failed on '{file_name}' section {section_index}: {reason}"
        )


class DocumentChunker:
    _HEADER_KEYS = ("h1", "h2", "h3", "h4", "h5", "h6")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self._pipeline = (
            Pipeline()
            .chunk_with("recursive", chunk_size=chunk_size)
            .refine_with("overlap", context_size=chunk_overlap)
        )

    async def chunk_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Chunks a list of section Documents.
        Returns (parents, children) where parents are the input docs enriched with
        header breadcrumb metadata, and children are the individual chunks.
        """
        all_parents: List[Document] = []
        all_children: List[Document] = []

        tasks = [self._chunk_single(doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for doc, res in zip(docs, results):
            if isinstance(res, Exception):
                logger.error(f"Failed to chunk '{doc.metadata.get('file_name', 'unknown')}' "
                             f"section {doc.metadata.get('section_index', '?')}: {res}")
            else:
                parents, children = res  # type: ignore
                all_parents.extend(parents)
                all_children.extend(children)

        logger.info(f"Chunked {len(docs)} sections → {len(all_children)} chunks")
        return all_parents, all_children

    async def _chunk_single(self, doc: Document) -> Tuple[List[Document], List[Document]]:
        """Chunks a single section Document into parent + child Documents."""
        try:
            parent_id = str(uuid.uuid4())

            breadcrumb_parts = [doc.metadata.get("file_name", "")]
            breadcrumb_parts.extend(
                doc.metadata[h] for h in self._HEADER_KEYS if doc.metadata.get(h)
            )
            breadcrumb = " > ".join(filter(None, breadcrumb_parts))

            parent_meta = {
                **doc.metadata,
                "parent_id": parent_id,
                "breadcrumb": breadcrumb,
            }
            parent = Document(page_content=doc.page_content, metadata=parent_meta)

            result = await self._pipeline.arun(doc.page_content)
            raw_chunks = result.chunks # type: ignore

            children = [
                Document(
                    page_content=chunk.text,
                    metadata={
                        **parent_meta,
                        "is_child": True,
                        "token_count": chunk.token_count,
                        "chunk_start": chunk.start_index,
                        "chunk_end": chunk.end_index,
                    },
                )
                for chunk in raw_chunks
                if chunk.text.strip()
            ]

            return [parent], children

        except Exception as e:
            raise ChunkingException(
                file_name=doc.metadata.get("file_name", "unknown"),
                section_index=doc.metadata.get("section_index", "?"),
                reason=str(e),
            ) from e
