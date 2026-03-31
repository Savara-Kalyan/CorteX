from pathlib import Path
from typing import List, Optional, Tuple

from .config import DocumentLoadingConfig
from .models import FileType
from .loader import FileTypeDetector
from ...logging import get_logger


logger = get_logger(__name__)


class DocumentLoadingValidator:
    """Validate input."""
    
    def __init__(self, config: DocumentLoadingConfig):
        """Initialize."""
        self.config = config
    
    def validate_directory(self, directory_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate directory."""
        
        if not directory_path.exists():
            error = f"Directory does not exist: {directory_path}"
            logger.warning("Validation failed", error=error)
            return False, error
        
        if not directory_path.is_dir():
            error = f"Path is not a directory: {directory_path}"
            logger.warning("Validation failed", error=error)
            return False, error
        
        return True, None
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate file."""
        
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        if not FileTypeDetector.is_supported(file_path):
            error = f"File type not supported: {file_path.suffix}"
            logger.warning("Validation failed", error=error)
            return False, error

        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > self.config.extraction.max_file_size_mb:
            error = f"File too large: {file_size_mb:.2f}MB (max: {self.config.extraction.max_file_size_mb}MB)"
            logger.warning("Validation failed", error=error)
            return False, error

        if file_size_bytes < self.config.extraction.min_file_size_bytes:
            error = f"File too small: {file_size_bytes} bytes"
            logger.warning("Validation failed", error=error)
            return False, error
        
        return True, None
    
    def validate_files(self, files: List[Path]) -> Tuple[List[Path], List[str]]:
        """Validate multiple files."""
        
        valid_files = []
        errors = []
        
        for file_path in files:
            is_valid, error_msg = self.validate_file(file_path)
            
            if is_valid:
                valid_files.append(file_path)
            else:
                errors.append(error_msg)
        
        logger.info(
            "File validation complete",
            total_files=len(files),
            valid_files=len(valid_files),
            errors=len(errors)
        )
        
        return valid_files, errors