import os
from enum import Enum
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


class Environment(Enum):
    """
    Environments.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ExtractionConfig:
    """
    Extraction settings.
    """
    
    char_count_threshold: int = 500
    max_concurrent_files: int = 5
    max_workers: int = 10
    extraction_timeout_seconds: float = 300.0
    enable_ocr: bool = True
    enable_vlm: bool = False
    vlm_model: str = "gpt-4-vision-preview"
    max_file_size_mb: int = 500
    min_file_size_bytes: int = 100


@dataclass
class DeduplicationConfig:
    """
    Deduplication settings.
    """
    
    enabled: bool = True
    method: str = "content_hash"  # content_hash, file_hash
    threshold: float = 0.95


@dataclass
class DocumentLoadingConfig:
    """
    Main config.
    """
    
    environment: Environment = Environment.PRODUCTION
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    
    documents_root: Path = Path("./documents")
    temp_dir: Path = Path("/tmp/cortex-doc-loading")
    
    enable_deduplication: bool = True
    enable_validation: bool = True
    
    @classmethod
    def from_env(cls) -> "DocumentLoadingConfig":
        """Load from environment."""
        return cls(
            environment=Environment(os.getenv("CORTEX_ENV", "production")),
            extraction=ExtractionConfig(
                char_count_threshold=int(os.getenv("DL_CHAR_THRESHOLD", "500")),
                max_concurrent_files=int(os.getenv("DL_MAX_CONCURRENT", "5")),
                enable_ocr=os.getenv("DL_ENABLE_OCR", "true").lower() == "true",
                enable_vlm=os.getenv("DL_ENABLE_VLM", "false").lower() == "true",
            ),
        )


def get_config() -> DocumentLoadingConfig:
    """
    Get config.
    """
    
    return DocumentLoadingConfig.from_env()