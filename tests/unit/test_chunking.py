import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.services.chunking.service import DocumentChunker, ChunkingException


# --- Fixtures & helpers ---

@pytest.fixture
def chunker():
    return DocumentChunker(chunk_size=512, chunk_overlap=50)


def _make_doc(content: str, **meta) -> Document:
    base = {"file_name": "test.pdf", "section_index": 0}
    return Document(page_content=content, metadata={**base, **meta})


def _make_chunk(text: str, tokens: int = 10, start: int = 0):
    chunk = MagicMock()
    chunk.text = text
    chunk.token_count = tokens
    chunk.start_index = start
    chunk.end_index = start + len(text)
    return chunk


def _make_pipeline_result(chunks):
    """Wraps a list of mock chunks in a pipeline result object (has .chunks)."""
    result = MagicMock()
    result.chunks = chunks
    return result


# --- ChunkingException ---

def test_chunking_exception_message():
    exc = ChunkingException(file_name="doc.pdf", section_index=2, reason="timeout")
    assert "doc.pdf" in str(exc)
    assert "2" in str(exc)
    assert "timeout" in str(exc)


# --- _chunk_single ---

@pytest.mark.asyncio
async def test_chunk_single_returns_parent_and_children(chunker):
    doc = _make_doc("Some content here.", h1="Introduction")
    mock_result = _make_pipeline_result([_make_chunk("Some content here.")])

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(return_value=mock_result)):
        parents, children = await chunker._chunk_single(doc)

    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].metadata["is_child"] is True
    assert children[0].metadata["token_count"] == 10


@pytest.mark.asyncio
async def test_chunk_single_breadcrumb_with_headers(chunker):
    doc = _make_doc("Content.", h1="Chapter", h2="Section")
    mock_result = _make_pipeline_result([_make_chunk("Content.")])

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(return_value=mock_result)):
        parents, _ = await chunker._chunk_single(doc)

    assert parents[0].metadata["breadcrumb"] == "test.pdf > Chapter > Section"


@pytest.mark.asyncio
async def test_chunk_single_breadcrumb_no_headers(chunker):
    doc = _make_doc("Plain text content.")
    mock_result = _make_pipeline_result([_make_chunk("Plain text content.")])

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(return_value=mock_result)):
        parents, _ = await chunker._chunk_single(doc)

    assert parents[0].metadata["breadcrumb"] == "test.pdf"


@pytest.mark.asyncio
async def test_chunk_single_skips_empty_chunks(chunker):
    doc = _make_doc("Content.")
    mock_result = _make_pipeline_result([_make_chunk("Content."), _make_chunk("   ")])

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(return_value=mock_result)):
        _, children = await chunker._chunk_single(doc)

    assert len(children) == 1


@pytest.mark.asyncio
async def test_chunk_single_raises_chunking_exception_on_failure(chunker):
    doc = _make_doc("Content.")

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(ChunkingException) as exc_info:
            await chunker._chunk_single(doc)

    assert "test.pdf" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


# --- chunk_documents ---

@pytest.mark.asyncio
async def test_chunk_documents_aggregates_results(chunker):
    docs = [_make_doc("Section one."), _make_doc("Section two.", section_index=1)]
    mock_result = _make_pipeline_result([_make_chunk("chunk")])

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(return_value=mock_result)):
        parents, children = await chunker.chunk_documents(docs)

    assert len(parents) == 2
    assert len(children) == 2


@pytest.mark.asyncio
async def test_chunk_documents_logs_and_continues_on_failure(chunker, caplog):
    docs = [_make_doc("Good content."), _make_doc("Bad content.", section_index=1)]
    good_result = _make_pipeline_result([_make_chunk("Good content.")])

    async def side_effect(text):
        if "Bad" in text:
            raise RuntimeError("parse error")
        return good_result

    with patch.object(chunker._pipeline, "arun", new=AsyncMock(side_effect=side_effect)):
        with caplog.at_level(logging.ERROR, logger="EnterpriseIngestion"):
            parents, children = await chunker.chunk_documents(docs)

    assert len(parents) == 1
    assert len(children) == 1
    assert any("test.pdf" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_chunk_documents_empty_input(chunker):
    parents, children = await chunker.chunk_documents([])
    assert parents == []
    assert children == []


# --- Integration-style tests ---

@pytest.mark.asyncio
async def test_single_document_chunking():
    """A long document is split and children retain metadata."""
    service = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "The pygmy hippo is a small hippopotamid native to the forests of West Africa. " * 10
    doc = Document(page_content=text, metadata={"source": "manual_test.pdf", "page": 1})

    _, children = await service.chunk_documents([doc])

    assert len(children) > 1
    assert children[0].metadata["source"] == "manual_test.pdf"
    assert children[0].metadata["page"] == 1
    assert "token_count" in children[0].metadata
    assert isinstance(children[0], Document)


@pytest.mark.asyncio
async def test_batch_document_chunking():
    """Multiple documents from different sources are all chunked."""
    service = DocumentChunker(chunk_size=100, chunk_overlap=20)
    docs = [
        Document(page_content="Content A " * 20, metadata={"id": "A"}),
        Document(page_content="Content B " * 20, metadata={"id": "B"}),
    ]

    _, children = await service.chunk_documents(docs)

    ids_found = {chunk.metadata["id"] for chunk in children}
    assert ids_found == {"A", "B"}
    assert len(children) >= 2


@pytest.mark.asyncio
async def test_small_document_no_split():
    """A document smaller than chunk_size stays as one chunk."""
    service = DocumentChunker(chunk_size=500, chunk_overlap=50)
    small_text = "Just a little bit of text."
    doc = Document(page_content=small_text, metadata={"type": "short"})

    _, children = await service.chunk_documents([doc])

    assert len(children) == 1
    assert children[0].page_content == small_text
