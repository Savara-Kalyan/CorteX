
import pytest
from pathlib import Path

from app.services.document_loading.validators import DocumentLoadingValidator
from app.services.document_loading.config import DocumentLoadingConfig, ExtractionConfig


class TestDocumentLoadingValidator:
    """Test document loading validator."""
    
    def test_validate_directory_exists(self, temp_dir):
        """Test validating existing directory."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        is_valid, error = validator.validate_directory(temp_dir)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_directory_not_exists(self):
        """Test validating non-existent directory."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        fake_path = Path("/nonexistent/directory/path")
        is_valid, error = validator.validate_directory(fake_path)
        
        assert is_valid is False
        assert "does not exist" in error
    
    def test_validate_directory_is_file(self, temp_dir):
        """Test validating when path is a file, not directory."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        # Create a file
        file_path = temp_dir / "file.txt"
        file_path.write_text("content")
        
        is_valid, error = validator.validate_directory(file_path)
        
        assert is_valid is False
        assert "not a directory" in error
    
    def test_validate_file_exists(self, sample_txt_path):
        """Test validating existing file."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        is_valid, error = validator.validate_file(sample_txt_path)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_file_too_large(self, temp_dir):
        """Test rejecting files that are too large."""
        config = DocumentLoadingConfig(
            extraction=ExtractionConfig(max_file_size_mb=1)  # 1MB max
        )
        validator = DocumentLoadingValidator(config)
        
        # Create a 2MB file
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (2 * 1024 * 1024))
        
        is_valid, error = validator.validate_file(large_file)
        
        assert is_valid is False
        assert "too large" in error
    
    def test_validate_file_unsupported_type(self, temp_dir):
        """Test rejecting unsupported file types."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        # Create an executable file
        exe_file = temp_dir / "program.exe"
        exe_file.write_text("binary content")
        
        is_valid, error = validator.validate_file(exe_file)
        
        assert is_valid is False
        assert "not supported" in error
    
    def test_validate_multiple_files(self, temp_dir):
        """Test validating multiple files."""
        config = DocumentLoadingConfig()
        validator = DocumentLoadingValidator(config)
        
        # Create multiple files
        valid_file = temp_dir / "valid.txt"
        valid_file.write_text("content " * 20)
        
        invalid_file = temp_dir / "invalid.exe"
        invalid_file.write_text("binary")
        
        files = [valid_file, invalid_file]
        valid_files, errors = validator.validate_files(files)
        
        assert len(valid_files) == 1
        assert len(errors) == 1
        assert valid_files[0] == valid_file