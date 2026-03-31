
import pytest
import asyncio
from pathlib import Path

from app.services.document_loading.service import DocumentLoadingService
from app.services.document_loading.models import FileType


class TestDocumentLoadingService:
    """Integration tests for DocumentLoadingService."""
    
    @pytest.mark.asyncio
    async def test_load_directory_with_files(self, document_loading_service, temp_dir):
        """Test loading documents from a directory with actual files."""
        
        # Create sample files
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("Sample document content. " * 50)
        
        # Load directory
        results, summary = await document_loading_service.load_directory(
            directory_path=temp_dir
        )
        
        # Assertions
        assert summary.total_files == 1
        assert summary.successfully_extracted == 1
        assert summary.failed_files == 0
        assert len(results) == 1
        assert results[0].documents  # Should have extracted documents
    
    @pytest.mark.asyncio
    async def test_load_directory_empty(self, document_loading_service, temp_dir):
        """Test loading from empty directory."""
        
        # Empty directory
        results, summary = await document_loading_service.load_directory(
            directory_path=temp_dir
        )
        
        assert summary.total_files == 0
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_load_directory_with_pattern(self, document_loading_service, temp_dir):
        """Test loading with file pattern."""
        
        # Create files
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("Content " * 50)
        
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_text("PDF content")
        
        # Load only txt files
        results, summary = await document_loading_service.load_directory(
            directory_path=temp_dir,
            pattern="*.txt"
        )
        
        # Should only get txt file
        assert summary.total_files == 1
        assert results[0].metrics.file_type == FileType.TXT
    
    @pytest.mark.asyncio
    async def test_load_directory_with_file_type_filter(
        self,
        document_loading_service,
        temp_dir
    ):
        """Test loading with file type filter."""
        
        # Create files
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("Content " * 50)
        
        md_file = temp_dir / "readme.md"
        md_file.write_text("# Markdown " * 50)
        
        # Load only txt files
        results, summary = await document_loading_service.load_directory(
            directory_path=temp_dir,
            file_types=[FileType.TXT]
        )
        
        # Should only get txt file
        assert summary.total_files == 1
        assert results[0].metrics.file_type == FileType.TXT
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_config, temp_dir):
        """Test that concurrent processing works."""
        
        # Create multiple files
        for i in range(5):
            file = temp_dir / f"document{i}.txt"
            file.write_text(f"Content {i} " * 50)
        
        config = test_config
        service = DocumentLoadingService(config=config)
        
        results, summary = await service.load_directory(
            directory_path=temp_dir,
            max_concurrent=2  # Test with 2 concurrent
        )
        
        # Should process all files
        assert summary.total_files == 5
        assert summary.successfully_extracted == 5
        
        service.shutdown()
    
    @pytest.mark.asyncio
    async def test_deduplication(self, document_loading_service, temp_dir):
        """Test that duplicates are removed."""
        
        # Create duplicate files
        content = "Sample content " * 50
        
        file1 = temp_dir / "file1.txt"
        file1.write_text(content)
        
        file2 = temp_dir / "file2.txt"
        file2.write_text(content)  # Same content
        
        results, summary = await document_loading_service.load_directory(
            directory_path=temp_dir
        )
        
        # Should have 2 files but only 1 in results (dedup)
        assert summary.total_files == 2
        assert len(results) == 1  # One removed as duplicate