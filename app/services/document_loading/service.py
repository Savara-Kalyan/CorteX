import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .config import DocumentLoadingConfig, get_config
from .models import (
    LoadingTask,
    LoadingStatus,
    LoadingSummary,
    ExtractionResult,
    FileType,
)
from .extractors import DocumentExtractor
from .validators import DocumentLoadingValidator
from .deduplication import DeduplicationManager
from .loader import FileTypeDetector
from .exceptions import ValidationException
from ...logging import get_logger


logger = get_logger(__name__)


class DocumentLoadingService:
    """Main document loading service with async + parallel processing.
    
    Features:
    - Async/await throughout
    - Parallel processing with controlled concurrency
    - Detailed latency tracking (time to first result, per-strategy timing)
    - Comprehensive logging and metrics
    """
    
    def __init__(self, config: Optional[DocumentLoadingConfig] = None):
        """Initialize service."""
        
        self.config = config or get_config()
        self.extractor = DocumentExtractor(self.config)
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.extraction.max_workers
        )
        
        self.validator = DocumentLoadingValidator(self.config)
        self.dedup_manager = DeduplicationManager(self.config.deduplication)
        
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "DocumentLoadingService initialized",
            environment=self.config.environment.value,
            max_concurrent_files=self.config.extraction.max_concurrent_files,
            max_workers=self.config.extraction.max_workers,
            deduplication_enabled=self.config.enable_deduplication,
        )
    
    def _discover_files(
        self,
        root_path: Path,
        pattern: Optional[str] = None,
        file_types: Optional[List[FileType]] = None
    ) -> List[Path]:
        """Discover files to process."""
        
        try:
            is_valid, error_msg = self.validator.validate_directory(root_path)
            if not is_valid:
                raise ValidationException(error_msg)
            
            if pattern:
                files = list(root_path.glob(pattern))
            else:
                files = list(root_path.rglob("*"))
            
            supported_files = [
                f for f in files
                if f.is_file() and FileTypeDetector.is_supported(f)
            ]
            
            if file_types:
                supported_files = [
                    f for f in supported_files
                    if FileTypeDetector.detect(f) in file_types
                ]
            
            valid_files, errors = self.validator.validate_files(supported_files)
            
            if errors:
                logger.warning(
                    "Validation errors",
                    total_errors=len(errors),
                    valid_files=len(valid_files)
                )
            
            logger.info(
                "File discovery completed",
                total_files=len(valid_files),
                root_path=str(root_path)
            )
            
            return valid_files
        
        except Exception as e:
            logger.error("File discovery failed", error=str(e))
            raise
    
    async def load_directory(
        self,
        directory_path: Path,
        pattern: Optional[str] = None,
        file_types: Optional[List[FileType]] = None,
        max_concurrent: Optional[int] = None,
    ) -> Tuple[List[ExtractionResult], LoadingSummary]:
        """Load all documents from directory.
        
        Uses async/parallel processing for speed.
        """
        
        task = LoadingTask(
            directory_path=directory_path,
            file_pattern=pattern,
            file_types=file_types,
        )
        
        return await self.load_with_task(task, max_concurrent)
    
    async def load_with_task(
        self,
        task: LoadingTask,
        max_concurrent: Optional[int] = None
    ) -> Tuple[List[ExtractionResult], LoadingSummary]:
        """Load documents with task tracking and latency metrics."""
        
        task.status = LoadingStatus.VALIDATING
        task_start_time = time.time()
        task.started_at = datetime.utcnow()
        
        # Latency tracking variables
        first_result_time = None
        extraction_times = []
        
        try:
            # Validate directory
            is_valid, error_msg = self.validator.validate_directory(task.directory_path)
            if not is_valid:
                raise ValidationException(error_msg)
            
            logger.info(
                "Task started",
                task_id=task.task_id,
                directory=str(task.directory_path)
            )
            
            # Discover files
            task.status = LoadingStatus.PROCESSING
            files = self._discover_files(
                task.directory_path,
                task.file_pattern,
                task.file_types
            )
            
            if not files:
                logger.warning(
                    "No supported files found",
                    task_id=task.task_id,
                    directory=str(task.directory_path)
                )
            
            # ============================================
            # ASYNC/PARALLEL PROCESSING WITH CONCURRENCY CONTROL
            # ============================================
            max_concurrent = max_concurrent or self.config.extraction.max_concurrent_files
            semaphore = asyncio.Semaphore(max_concurrent)
            
            extraction_start_time = time.time()
            
            async def extract_with_limit(file_path: Path):
                """Extract one file, respecting concurrency limit.
                
                This ensures we process max_concurrent files simultaneously,
                not more.
                """
                nonlocal first_result_time
                
                async with semaphore:  # Wait for available slot
                    file_extraction_start = time.time()
                    result = await self.extractor.extract_file(file_path, self.executor)
                    file_extraction_time = time.time() - file_extraction_start
                    
                    # Track first result time (latency metric)
                    if first_result_time is None:
                        first_result_time = time.time() - task_start_time
                        logger.info(
                            "First file completed",
                            task_id=task.task_id,
                            latency_seconds=f"{first_result_time:.3f}",
                            file_path=str(file_path),
                            strategy=result.metrics.extraction_strategy.value
                        )
                    
                    extraction_times.append(file_extraction_time)
                    return result
            
            # Create tasks for all files (async, not executed yet)
            tasks = [extract_with_limit(f) for f in files]
            
            # Run all tasks concurrently (this is where the magic happens!)
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            extraction_total_time = time.time() - extraction_start_time
            
            # Filter successful extractions
            extraction_results = [
                r for r in results 
                if isinstance(r, ExtractionResult)
            ]
            
            # ============================================
            # DEDUPLICATION (sequential)
            # ============================================
            if self.config.enable_deduplication:
                dedup_start = time.time()
                deduped_results = []
                
                for result in extraction_results:
                    is_dup, original = self.dedup_manager.is_duplicate(result)
                    
                    if not is_dup:
                        deduped_results.append(result)
                    else:
                        logger.info(
                            "Duplicate skipped",
                            file_path=str(result.file_path),
                            original_file=str(original)
                        )
                
                extraction_results = deduped_results
                dedup_time = time.time() - dedup_start
                
                logger.debug(
                    "Deduplication completed",
                    task_id=task.task_id,
                    dedup_time_seconds=f"{dedup_time:.3f}"
                )
            
            # ============================================
            # FINAL SUMMARY
            # ============================================
            task.results = extraction_results
            task.status = LoadingStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            summary = self._create_summary(extraction_results, total_discovered=len(files))
            
            total_duration = time.time() - task_start_time
            throughput = len(files) / total_duration if total_duration > 0 else 0
            
            logger.info(
                "Task completed",
                task_id=task.task_id,
                total_files=len(files),
                successfully_extracted=summary.successfully_extracted,
                failed_files=summary.failed_files,
                total_documents=summary.total_documents,
                total_chars=summary.total_chars,
                success_rate=f"{summary.success_rate:.2%}",
                extraction_time_seconds=f"{extraction_total_time:.3f}",
                total_time_seconds=f"{total_duration:.3f}",
                time_to_first_result_seconds=f"{first_result_time:.3f}" if first_result_time else "N/A",
                throughput_files_per_sec=f"{throughput:.2f}",
                avg_extraction_time_seconds=f"{summary.avg_extraction_time:.3f}",
                avg_chars_per_file=f"{summary.avg_chars_per_file:.0f}",
                files_using_fallback=summary.files_using_fallback,
                extraction_strategies=summary.extraction_strategies,
                max_concurrent_files=max_concurrent
            )
            
            return extraction_results, summary
        
        except Exception as e:
            logger.error(
                "Task failed",
                task_id=task.task_id,
                error=str(e)
            )
            
            task.status = LoadingStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            raise
    
    def _create_summary(self, results: List[ExtractionResult], total_discovered: int = 0) -> LoadingSummary:
        """Create high-level summary statistics."""

        total_files = total_discovered if total_discovered > 0 else len(results)
        successfully_extracted = sum(1 for r in results if r.documents)
        failed_files = total_files - successfully_extracted
        total_documents = sum(len(r.documents) for r in results)
        total_chars = sum(r.metrics.char_count for r in results)
        total_time = sum(r.metrics.extraction_time_seconds for r in results)
        files_using_fallback = sum(1 for r in results if r.used_fallback)
        
        # Count by strategy
        strategies = {}
        for r in results:
            strategy = r.metrics.extraction_strategy.value
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return LoadingSummary(
            total_files=total_files,
            successfully_extracted=successfully_extracted,
            failed_files=failed_files,
            total_documents=total_documents,
            total_chars=total_chars,
            total_time_seconds=total_time,
            files_using_fallback=files_using_fallback,
            extraction_strategies=strategies,
        )
    
    def shutdown(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self.dedup_manager.reset()
        logger.info("DocumentLoadingService shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()