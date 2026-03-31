import pytest
import os
from pathlib import Path

from app.logging import (
    LogLevel
)

from app.services.document_loading.config import (
    DocumentLoadingConfig,
    ExtractionConfig,
)


class TestExtractionConfig:
    """Test extraction configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        
        assert config.char_count_threshold == 500
        assert config.max_concurrent_files == 5
        assert config.enable_ocr is True
        assert config.enable_vlm is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExtractionConfig(
            char_count_threshold=1000,
            max_concurrent_files=10,
        )
        
        assert config.char_count_threshold == 1000
        assert config.max_concurrent_files == 10
    
    def test_validation(self):
        """Test invalid configuration."""
        # Negative threshold should still work (Python dataclass doesn't validate)
        config = ExtractionConfig(char_count_threshold=-1)
        assert config.char_count_threshold == -1


class TestDocumentLoadingConfig:
    """Test main configuration."""
    
    def test_from_env_with_defaults(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("CORTEX_ENV", "development")
        monkeypatch.setenv("DL_CHAR_THRESHOLD", "1000")
        monkeypatch.setenv("DL_MAX_CONCURRENT", "20")
        
        config = DocumentLoadingConfig.from_env()
        
        assert config.environment.value == "development"
        assert config.extraction.char_count_threshold == 1000
        assert config.extraction.max_concurrent_files == 20
    
    def test_from_env_with_defaults_fallback(self, monkeypatch):
        """Test default values when env vars not set."""
        # Clear any existing env vars
        monkeypatch.delenv("CORTEX_ENV", raising=False)
        monkeypatch.delenv("DL_CHAR_THRESHOLD", raising=False)
        
        config = DocumentLoadingConfig.from_env()
        
        assert config.environment.value == "production"
        assert config.extraction.char_count_threshold == 500