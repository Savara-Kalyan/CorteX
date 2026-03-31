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
                all_sections.extend(res)

        logger.info(f"Total sections loaded: {len(all_sections)}")
        return all_sections

    async def process_single_file(self, file_path: Path) -> List[Document]:
        """
        Converts a single file to markdown via Docling, then splits it into
        one LangChain Document per markdown section (heading block).
        """
        t_start = time.perf_counter()
        ingestion_timestamp = datetime.now().isoformat()

        try:
            loader = DoclingLoader(
                file_path=str(file_path),
                export_type=self.export_type,
            )

            # Docling conversion is CPU/IO intensive — offload to thread
            raw_docs = await asyncio.to_thread(loader.load)

            if not raw_docs:
                raise ExtractionException(file_path.name, "Docling returned no content")

            full_markdown = raw_docs[0].page_content
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
                        **section.metadata,
                    },
                ))

            elapsed = time.perf_counter() - t_start
            logger.info(f"Processed '{file_path.name}': {len(documents)} sections in {elapsed:.2f}s")
            return documents

        except ExtractionException:
            raise
        except Exception as e:
            raise ExtractionException(file_path.name, str(e)) from e

