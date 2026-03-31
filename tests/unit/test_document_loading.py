import pytest
from unittest.mock import patch

from langchain_core.documents import Document

from app.services.document_loading.service import (
    DocumentLoader,
    FileNotFoundException,
    ExtractionException,
)


@pytest.fixture
def loader():
    return DocumentLoader()


# --- _md_splitter.split_text ---

def test_split_no_headings(loader):
    sections = loader._md_splitter.split_text("Just plain text.")
    assert len(sections) == 1
    assert sections[0].page_content == "Just plain text."
    assert sections[0].metadata == {}


def test_split_preamble_and_headings(loader):
    md = "Intro text.\n\n## Section A\nContent A.\n\n## Section B\nContent B."
    sections = loader._md_splitter.split_text(md)
    assert sections[0].page_content == "Intro text."
    assert sections[0].metadata == {}
    assert sections[1].metadata.get("h2") == "Section A"
    assert "Content A." in sections[1].page_content
    assert sections[2].metadata.get("h2") == "Section B"


def test_split_nested_headings(loader):
    md = "# Title\nIntro.\n\n## Sub\nBody."
    sections = loader._md_splitter.split_text(md)
    assert sections[0].metadata.get("h1") == "Title"
    assert sections[1].metadata.get("h2") == "Sub"


# --- load_directory ---

@pytest.mark.asyncio
async def test_load_directory_invalid_path(loader):
    with pytest.raises(FileNotFoundException):
        await loader.load_directory("/nonexistent/path")


@pytest.mark.asyncio
async def test_load_directory_empty(loader, tmp_path):
    docs = await loader.load_directory(str(tmp_path))
    assert docs == []


# --- process_single_file ---

@pytest.mark.asyncio
async def test_process_single_file_returns_one_doc_per_section(loader, tmp_path):
    md_content = "# Section 1\nContent one.\n\n# Section 2\nContent two."
    mock_doc = Document(page_content=md_content, metadata={})

    with patch("app.services.document_loading.service.DoclingLoader") as MockLoader:
        MockLoader.return_value.load.return_value = [mock_doc]
        fake_file = tmp_path / "test.pdf"
        fake_file.touch()

        docs = await loader.process_single_file(fake_file)

    assert len(docs) == 2
    assert docs[0].metadata["section_heading"] == "Section 1"
    assert docs[1].metadata["section_heading"] == "Section 2"
    assert docs[0].metadata["section_index"] == 0
    assert docs[1].metadata["section_index"] == 1


@pytest.mark.asyncio
async def test_process_single_file_metadata_fields(loader, tmp_path):
    md_content = "## Only Section\nSome content here."
    mock_doc = Document(page_content=md_content, metadata={})

    with patch("app.services.document_loading.service.DoclingLoader") as MockLoader:
        MockLoader.return_value.load.return_value = [mock_doc]
        fake_file = tmp_path / "report.pdf"
        fake_file.touch()

        docs = await loader.process_single_file(fake_file)

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["file_name"] == "report.pdf"
    assert meta["file_type"] == "pdf"
    assert meta["source"] == str(fake_file)
    assert "section_hash" in meta
    assert "ingestion_timestamp" in meta


@pytest.mark.asyncio
async def test_process_single_file_empty_extraction_raises(loader, tmp_path):
    with patch("app.services.document_loading.service.DoclingLoader") as MockLoader:
        MockLoader.return_value.load.return_value = []
        fake_file = tmp_path / "empty.pdf"
        fake_file.touch()

        with pytest.raises(ExtractionException):
            await loader.process_single_file(fake_file)
