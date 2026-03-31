
import pytest
from pathlib import Path

from app.services.document_loading.loader import FileTypeDetector
from app.services.document_loading.models import FileType


class TestFileTypeDetector:
    """Test file type detection."""
    
    def test_detect_pdf(self, temp_dir):
        """Test detecting PDF files."""
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(pdf_file)
        
        assert file_type == FileType.PDF
    
    def test_detect_txt(self, temp_dir):
        """Test detecting TXT files."""
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(txt_file)
        
        assert file_type == FileType.TXT
    
    def test_detect_docx(self, temp_dir):
        """Test detecting DOCX files."""
        docx_file = temp_dir / "document.docx"
        docx_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(docx_file)
        
        assert file_type == FileType.DOCX
    
    def test_detect_markdown(self, temp_dir):
        """Test detecting Markdown files."""
        md_file = temp_dir / "document.md"
        md_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(md_file)
        
        assert file_type == FileType.MARKDOWN
    
    def test_detect_unsupported(self, temp_dir):
        """Test detecting unsupported file type."""
        exe_file = temp_dir / "program.exe"
        exe_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(exe_file)
        
        assert file_type == FileType.UNKNOWN
    
    def test_is_supported_yes(self, temp_dir):
        """Test checking if supported file type."""
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_text("dummy")
        
        is_supported = FileTypeDetector.is_supported(pdf_file)
        
        assert is_supported is True
    
    def test_is_supported_no(self, temp_dir):
        """Test checking if unsupported file type."""
        exe_file = temp_dir / "program.exe"
        exe_file.write_text("dummy")
        
        is_supported = FileTypeDetector.is_supported(exe_file)
        
        assert is_supported is False
    
    def test_case_insensitive(self, temp_dir):
        """Test that detection is case-insensitive."""
        pdf_file = temp_dir / "DOCUMENT.PDF"
        pdf_file.write_text("dummy")
        
        file_type = FileTypeDetector.detect(pdf_file)
        
        assert file_type == FileType.PDF