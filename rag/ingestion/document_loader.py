from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

_HEADERS = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ("####", "h4"), ("#####", "h5"), ("######", "h6"),
]


class DocumentLoader:
    def __init__(self):
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS,
            strip_headers=False,
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        files = list(path.rglob("*.md"))
        logger.info("Found %d markdown files in %s", len(files), dir_path)

        docs: List[Document] = []
        for f in files:
            docs.extend(self._load_file(f))

        logger.info("Loaded %d sections from %d files", len(docs), len(files))
        return docs

    def _load_file(self, file_path: Path) -> List[Document]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        sections = self._splitter.split_text(text)
        domain = file_path.parent.name
        return [
            Document(
                page_content=s.page_content.strip(),
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "domain": domain,
                    **s.metadata,
                },
            )
            for s in sections
            if s.page_content.strip()
        ]


DocumentIngestionService = DocumentLoader
