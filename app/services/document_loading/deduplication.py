import hashlib
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import DeduplicationConfig
from .models import ExtractionResult
from ...logging import get_logger


logger = get_logger(__name__)


class DeduplicationManager:
    """Manage deduplication."""
    
    def __init__(self, config: DeduplicationConfig):
        """Initialize."""
        self.config = config
        self.seen_hashes: Dict[str, Path] = {}
    
    def get_content_hash(self, documents: List) -> str:
        """Get hash of content."""
        
        combined_content = "".join(doc.page_content for doc in documents)
        return hashlib.sha256(combined_content.encode()).hexdigest()
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file."""
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def is_duplicate(self, result: ExtractionResult) -> Tuple[bool, Optional[Path]]:
        """Check if duplicate."""
        
        if not self.config.enabled:
            return False, None
        
        try:
            if self.config.method == "content_hash":
                content_hash = self.get_content_hash(result.documents)
            else:
                content_hash = self.get_file_hash(result.file_path)
            
            if content_hash in self.seen_hashes:
                original_path = self.seen_hashes[content_hash]
                logger.info(
                    "Duplicate detected",
                    file_path=str(result.file_path),
                    original_file=str(original_path),
                )
                return True, original_path
            
            self.seen_hashes[content_hash] = result.file_path
            return False, None
        
        except Exception as e:
            logger.warning("Deduplication check failed", error=str(e))
            return False, None
    
    def reset(self):
        """Reset state."""
        self.seen_hashes.clear()
        logger.debug("Deduplication state reset")