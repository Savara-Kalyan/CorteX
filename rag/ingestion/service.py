import time
import asyncio
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List

from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from settings.config import settings

logger = logging.getLogger("EnterpriseIngestion")


class DocumentLoadingException(Exception):
    def __init__(self, message: str):
        super().__init__(f"[DocumentLoading] {message}")


class FileNotFoundException(DocumentLoadingException):
    def __init__(self, path: str):
        super().__init__(f"Directory not found or is not a valid directory: '{path}'")


class ExtractionException(DocumentLoadingException):
    def __init__(self, file_name: str, reason: str):
        super().__init__(f"Failed to extract content from '{file_name}': {reason}")


def _extract_with_unstructured(file_path: str) -> str:
    """Extract text using unstructured.io."""
    from unstructured.partition.auto import partition

    elements = partition(file_path)
    return "\n".join(str(el) for el in elements)


def _extract_with_docling(file_path: str, export_type: ExportType) -> str:
    """Extract content using Docling (OCR-capable)."""
    loader = DoclingLoader(file_path=file_path, export_type=export_type)
    raw_docs = loader.load()
    if not raw_docs:
        raise ExtractionException(Path(file_path).name, "Docling returned no content")
    return raw_docs[0].page_content


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".ascii", ".md"}

    _HEADERS_TO_SPLIT_ON = [
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
        ("####", "h4"), ("#####", "h5"), ("######", "h6"),
    ]

    def __init__(self):
        self.export_type = ExportType.MARKDOWN
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

    async def load_directory(self, dir_path: str) -> List[Document]:
        """Scans a directory and processes all supported files in parallel."""
        path = Path(dir_path)
        if not path.is_dir():
            raise FileNotFoundException(dir_path)

        files = [f for f in path.rglob("*") if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        logger.info(f"Found {len(files)} files for ingestion.")

        tasks = [self.process_single_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_sections: List[Document] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Task failure: {res}")
            else:
                all_sections.extend(res)  # type: ignore[arg-type]

        logger.info(f"Total sections loaded: {len(all_sections)}")
        return all_sections

    async def _smart_extract(self, file_path: Path) -> tuple[str, str]:
        """
        Smart extraction strategy:
        1. Try unstructured on the first N sample pages.
        2. If content is sparse (< CHAR_COUNT_THRESHOLD chars), fall back to Docling.
        Returns (markdown_content, extractor_type).
        """
        file_str = str(file_path)

        try:
            content = await asyncio.to_thread(_extract_with_unstructured, file_str)
            if len(content) >= settings.ingestion.char_count_threshold:
                logger.info(f"Unstructured succeeded for '{file_path.name}' ({len(content)} chars)")
                return content, "unstructured"
            logger.info(
                f"Unstructured content sparse for '{file_path.name}' "
                f"({len(content)} chars < {settings.ingestion.char_count_threshold}), falling back to Docling"
            )
        except Exception as e:
            logger.warning(f"Unstructured failed for '{file_path.name}': {e}. Falling back to Docling.")

        content = await asyncio.to_thread(_extract_with_docling, file_str, self.export_type)
        logger.info(f"Docling extraction succeeded for '{file_path.name}'")
        return content, "docling"

    async def process_single_file(self, file_path: Path) -> List[Document]:
        """
        Extracts a file using smart extraction (unstructured → Docling fallback),
        splits the markdown into sections, and attaches metadata per section.
        """
        t_start = time.perf_counter()
        ingestion_timestamp = datetime.now().isoformat()

        try:
            full_markdown, extractor_type = await self._smart_extract(file_path)
            splits = await asyncio.to_thread(self._md_splitter.split_text, full_markdown)

            documents: List[Document] = []
            for i, section in enumerate(splits):
                text = section.page_content.strip()
                if not text:
                    continue

                # Deepest header level present in this section's metadata
                section_heading = next(
                    (section.metadata[h] for h in ("h6", "h5", "h4", "h3", "h2", "h1")
                     if h in section.metadata),
                    "",
                )
                section_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower().lstrip("."),
                        "section_index": i,
                        "section_heading": section_heading,
                        "section_hash": section_hash,
                        "ingestion_timestamp": ingestion_timestamp,
                        "text_len_chars": len(text),
                        "extractor_type": extractor_type,
                        **section.metadata,
                    },
                ))

            elapsed = time.perf_counter() - t_start
            logger.info(
                f"Processed '{file_path.name}': {len(documents)} sections "
                f"via {extractor_type} in {elapsed:.2f}s"
            )
            return documents

        except ExtractionException:
            raise
        except Exception as e:
            raise ExtractionException(file_path.name, str(e)) from e

