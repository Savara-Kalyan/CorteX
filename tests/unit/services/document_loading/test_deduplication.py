import pytest
from langchain_core.documents import Document

from app.services.document_loading.deduplication import DeduplicationManager
from app.services.document_loading.config import DeduplicationConfig
from app.services.document_loading.models import ExtractionResult


class TestDeduplicationManager:
    """Test deduplication manager."""
    
    def test_get_content_hash(self, sample_document):
        """Test content hash generation."""
        config = DeduplicationConfig(enabled=True)
        manager = DeduplicationManager(config)
        
        hash1 = manager.get_content_hash([sample_document])
        hash2 = manager.get_content_hash([sample_document])
        
        # Same content = same hash
        assert hash1 == hash2
    
    def test_get_content_hash_different(self, sample_document):
        """Test different content produces different hash."""
        config = DeduplicationConfig(enabled=True)
        manager = DeduplicationManager(config)
        
        doc1 = Document(page_content="Content A")
        doc2 = Document(page_content="Content B")
        
        hash1 = manager.get_content_hash([doc1])
        hash2 = manager.get_content_hash([doc2])
        
        # Different content = different hash
        assert hash1 != hash2
    
    def test_is_duplicate_first_time(self, sample_extraction_result):
        """Test that first occurrence is not marked as duplicate."""
        config = DeduplicationConfig(enabled=True)
        manager = DeduplicationManager(config)
        
        is_dup, original = manager.is_duplicate(sample_extraction_result)
        
        assert is_dup is False
        assert original is None
    
    def test_is_duplicate_second_time(self, sample_extraction_result):
        """Test that second occurrence is marked as duplicate."""
        config = DeduplicationConfig(enabled=True)
        manager = DeduplicationManager(config)
        
        # First time
        is_dup1, _ = manager.is_duplicate(sample_extraction_result)
        assert is_dup1 is False
        
        # Second time (same content)
        is_dup2, original = manager.is_duplicate(sample_extraction_result)
        assert is_dup2 is True
        assert original == sample_extraction_result.file_path
    
    def test_deduplication_disabled(self, sample_extraction_result):
        """Test that deduplication can be disabled."""
        config = DeduplicationConfig(enabled=False)
        manager = DeduplicationManager(config)
        
        is_dup1, _ = manager.is_duplicate(sample_extraction_result)
        is_dup2, _ = manager.is_duplicate(sample_extraction_result)
        
        # Never duplicates when disabled
        assert is_dup1 is False
        assert is_dup2 is False
    
    def test_reset(self, sample_extraction_result):
        """Test that reset clears deduplication state."""
        config = DeduplicationConfig(enabled=True)
        manager = DeduplicationManager(config)
        
        # First occurrence
        is_dup1, _ = manager.is_duplicate(sample_extraction_result)
        assert is_dup1 is False
        
        # Reset
        manager.reset()
        
        # Should be treated as new again
        is_dup2, _ = manager.is_duplicate(sample_extraction_result)
        assert is_dup2 is False