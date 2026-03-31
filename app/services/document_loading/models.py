from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid
from langchain_core.documents import Document


class FileType(Enum):
    """
    Supported file types.
    """

    PDF = "pdf"
    MARKDOWN = "md"
    DOCX = "docx"
    TXT = "txt"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    UNKNOWN = "unknown"


class ExtractionStrategy(Enum):
    """
    Extraction strategies.
    """

    STANDARD = "standard"    # PDFPlumber
    DOCLING = "docling"      # Docling
    OCR = "ocr"              # Last resort
    VLM = "vlm"              # Desperate measures
    FAILED = "failed"


class LoadingStatus(Enum):
    """
    Task status.
    """

    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractionMetrics:
    """
    Metrics from extraction.
    """
    
    file_path: Path
    file_type: FileType
    file_size_bytes: int
    extraction_strategy: ExtractionStrategy
    char_count: int = 0
    page_count: int = 0
    extraction_time_seconds: float = 0.0
    error_message: Optional[str] = None
    document_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        data["file_type"] = self.file_type.value
        data["extraction_strategy"] = self.extraction_strategy.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class ExtractionResult:
    """
    Result from extraction.
    """
    
    file_path: Path
    documents: List[Document]
    metrics: ExtractionMetrics
    used_fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict.
        """

        return {
            "file_path": str(self.file_path),
            "documents": [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
            "metrics": self.metrics.to_dict(),
            "used_fallback": self.used_fallback,
        }


@dataclass
class LoadingTask:
    """
    Loading task.
    """
    
    task_id: str = field(default_factory= lambda: str(uuid.uuid4()))
    directory_path: Path = Path("./documents")
    status: LoadingStatus = LoadingStatus.PENDING
    
    file_pattern: Optional[str] = None
    file_types: Optional[List[FileType]] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    results: List[ExtractionResult] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "task_id": self.task_id,
            "directory_path": str(self.directory_path),
            "status": self.status.value,
            "file_pattern": self.file_pattern,
            "file_types": [ft.value for ft in (self.file_types or [])],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_count": len(self.results),
        }


@dataclass
class LoadingSummary:
    """Summary of loading."""
    
    total_files: int
    successfully_extracted: int
    failed_files: int
    total_documents: int
    total_chars: int
    total_time_seconds: float
    files_using_fallback: int
    extraction_strategies: Dict[str, int]
    
    success_rate: float = 0.0
    avg_extraction_time: float = 0.0
    avg_chars_per_file: float = 0.0
    
    def __post_init__(self):
        """Calculate metrics."""
        if self.total_files > 0:
            self.success_rate = self.successfully_extracted / self.total_files
            self.avg_extraction_time = self.total_time_seconds / self.total_files
            self.avg_chars_per_file = self.total_chars / self.total_files if self.total_chars > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)