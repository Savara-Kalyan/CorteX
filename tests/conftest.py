import pytest
from pathlib import Path
import tempfile
from langchain_core.documents import Document

from app.services.document_loading.config import DocumentLoadingConfig, ExtractionConfig
from app.services.document_loading.service import DocumentLoadingService


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF file for testing."""
    
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_text("Sample PDF content")
    return pdf_path


@pytest.fixture
def sample_txt_path(temp_dir):
    """Create a sample TXT file for testing."""

    txt_path = temp_dir / "sample.txt"
    txt_path.write_text("Sample text content. " * 10)
    return txt_path


@pytest.fixture
def test_config():
    """Create test configuration (minimal, fast)."""

    return DocumentLoadingConfig(
        extraction=ExtractionConfig(
            char_count_threshold=100,
            max_concurrent_files=2,
            max_workers=2,
            enable_ocr=False,  # Disable slow operations
            enable_vlm=False,
        ),
    )


@pytest.fixture
def document_loading_service(test_config, temp_dir):
    """Create document loading service for testing."""
    service = DocumentLoadingService(config=test_config)
    yield service
    service.shutdown()


@pytest.fixture
def sample_document():
    """Create a sample LangChain Document."""
    return Document(
        page_content="This is sample content for testing. " * 10,
        metadata={"source": "test", "file_type": "pdf"}
    )


@pytest.fixture
def sample_extraction_result(sample_pdf_path, sample_document):
    """Create a sample extraction result."""
    from app.services.document_loading.models import (
        ExtractionResult,
        ExtractionMetrics,
        FileType,
        ExtractionStrategy,
    )
    
    metrics = ExtractionMetrics(
        file_path=sample_pdf_path,
        file_type=FileType.PDF,
        file_size_bytes=1024,
        extraction_strategy=ExtractionStrategy.STANDARD,
        char_count=len(sample_document.page_content),
        page_count=1,
        extraction_time_seconds=0.5,
    )
    
    return ExtractionResult(
        file_path=sample_pdf_path,
        documents=[sample_document],
        metrics=metrics,
        used_fallback=False,
    )