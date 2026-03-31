import asyncio
import time
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document

from .models import (
    FileType,
    ExtractionStrategy,
    ExtractionMetrics,
    ExtractionResult,
)
from .loader import (
    FileTypeDetector,
    StandardLoader,
    DoclingLoader,
    OCRLoader,
    VLMLoader,
)
from .analyzers import ContentQualityAnalyzer
from .exceptions import ExtractionException
from .config import DocumentLoadingConfig
from ...logging import get_logger


logger = get_logger(__name__)


class DocumentExtractor:
    """Intelligent document extractor with smart fallback strategy.
    
    Strategy order (optimized for speed and cost):
    1. StandardLoader (PDFPlumber) - Fast, 99% accurate for text PDFs
    2. DoclingLoader - Smart, handles both text and scanned PDFs
    3. OCRLoader - Slower, less accurate, but works for images
    4. VLMLoader - Most powerful but expensive ($$$)
    """
    
    def __init__(self, config: DocumentLoadingConfig):
        """Initialize extractor."""
        self.config = config
        self.analyzer = ContentQualityAnalyzer(
            char_count_threshold=config.extraction.char_count_threshold
        )
        
        # Initialize VLM if enabled
        self.vlm_extractor = None
        if config.extraction.enable_vlm:
            try:
                self.vlm_extractor = VLMLoader(model=config.extraction.vlm_model)
            except ExtractionException as e:
                logger.warning("VLM not available", error=e.message)
    
    async def extract_file(
        self,
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> ExtractionResult:
        """Extract single file with detailed timing.
        
        Tracks:
        - Total extraction time
        - Time for each strategy (Standard, Docling, OCR, VLM)
        - Which strategy succeeded
        - Character count and quality metrics
        """
        
        file_start_time = time.time()
        
        file_type = FileTypeDetector.detect(file_path)
        file_size = file_path.stat().st_size
        
        logger.info(
            "Processing file",
            file_path=str(file_path),
            file_type=file_type.value,
            file_size_bytes=file_size
        )
        
        # ============================================
        # STRATEGY 1: Standard Extraction
        # ============================================
        standard_start_time = time.time()
        
        try:
            standard_docs, standard_error = await self._extract_standard(
                file_path, file_type, executor
            )
        except Exception as e:
            logger.error(
                "Standard extraction error",
                file_path=str(file_path),
                error=str(e)
            )
            standard_docs, standard_error = [], str(e)
        
        standard_elapsed = time.time() - standard_start_time
        
        
        char_count, should_fallback = self.analyzer.analyze(standard_docs)
        
        if not should_fallback:
            # Standard extraction was good enough!
            logger.debug(
                "Standard extraction sufficient",
                file_path=str(file_path),
                char_count=char_count,
                standard_time_seconds=f"{standard_elapsed:.3f}"
            )
            
            total_elapsed = time.time() - file_start_time
            
            metrics = ExtractionMetrics(
                file_path=file_path,
                file_type=file_type,
                file_size_bytes=file_size,
                extraction_strategy=ExtractionStrategy.STANDARD,
                char_count=char_count,
                page_count=len(standard_docs),
                extraction_time_seconds=total_elapsed,
                error_message=None,
                document_count=len(standard_docs),
            )
            
            logger.info(
                "File processing complete",
                file_path=str(file_path),
                strategy="standard",
                char_count=char_count,
                total_time_seconds=f"{total_elapsed:.3f}",
                standard_time_seconds=f"{standard_elapsed:.3f}",
                docling_time_seconds="N/A",
                ocr_time_seconds="N/A",
                vlm_time_seconds="N/A",
                used_fallback=False
            )
            
            return ExtractionResult(
                file_path=file_path,
                documents=standard_docs,
                metrics=metrics,
                used_fallback=False,
            )
        
        # ============================================
        # FALLBACK STRATEGY: Try Docling first
        # ============================================
        logger.info(
            "Content sparse from standard, trying fallback",
            file_path=str(file_path),
            char_count=char_count,
            threshold=self.config.extraction.char_count_threshold
        )
        
        final_docs, strategy, used_fallback, docling_elapsed, ocr_elapsed, vlm_elapsed = await self._extract_with_fallback(
            file_path, file_type, standard_docs, executor, char_count
        )
        
        # ============================================
        # Final Metrics & Logging
        # ============================================
        total_elapsed = time.time() - file_start_time
        final_char_count = sum(len(doc.page_content) for doc in final_docs)
        
        metrics = ExtractionMetrics(
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=file_size,
            extraction_strategy=strategy,
            char_count=final_char_count,
            page_count=len(final_docs),
            extraction_time_seconds=total_elapsed,
            error_message=standard_error if strategy == ExtractionStrategy.FAILED else None,
            document_count=len(final_docs),
        )
        
        logger.info(
            "File processing complete",
            file_path=str(file_path),
            strategy=strategy.value,
            char_count=final_char_count,
            total_time_seconds=f"{total_elapsed:.3f}",
            standard_time_seconds=f"{standard_elapsed:.3f}",
            docling_time_seconds=f"{docling_elapsed:.3f}" if docling_elapsed else "N/A",
            ocr_time_seconds=f"{ocr_elapsed:.3f}" if ocr_elapsed else "N/A",
            vlm_time_seconds=f"{vlm_elapsed:.3f}" if vlm_elapsed else "N/A",
            used_fallback=used_fallback
        )
        
        return ExtractionResult(
            file_path=file_path,
            documents=final_docs,
            metrics=metrics,
            used_fallback=used_fallback,
        )
    
    async def _extract_standard(
        self,
        file_path: Path,
        file_type: FileType,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> Tuple[List[Document], Optional[str]]:
        """Extract using standard loader."""
        
        try:
            docs = await StandardLoader.extract(file_path, file_type, executor)
            return docs, None
        except Exception as e:
            return [], str(e)
    
    async def _extract_with_fallback(
        self,
        file_path: Path,
        file_type: FileType,
        standard_docs: List[Document],
        executor: Optional[ThreadPoolExecutor] = None,
        standard_char_count: int = 0
    ) -> Tuple[List[Document], ExtractionStrategy, bool, float, float, float]:
        """Smart fallback strategy with detailed timing.
        
        Returns:
            (documents, strategy, used_fallback, docling_time, ocr_time, vlm_time)
        """
        
        docling_elapsed = 0.0
        ocr_elapsed = 0.0
        vlm_elapsed = 0.0
        
        # ============================================
        # FALLBACK 1: Docling 
        # ============================================
        if file_type == FileType.PDF:
            docling_start = time.time()
            docling_docs = await DoclingLoader.load_pdf_with_docling(file_path, executor)
            docling_elapsed = time.time() - docling_start
            
            docling_char_count = sum(len(doc.page_content) for doc in docling_docs)
            
            logger.info(
                "Docling attempted",
                file_path=str(file_path),
                docling_char_count=docling_char_count,
                docling_time_seconds=f"{docling_elapsed:.3f}",
                standard_char_count=standard_char_count,
                improvement=f"{docling_char_count - standard_char_count:+d} chars"
            )
            
            if docling_char_count > standard_char_count:
                logger.info(
                    "Docling improved extraction",
                    file_path=str(file_path),
                    docling_chars=docling_char_count,
                    docling_time_seconds=f"{docling_elapsed:.3f}",
                    standard_chars=standard_char_count
                )
                return docling_docs, ExtractionStrategy.DOCLING, True, docling_elapsed, 0.0, 0.0
        
        # ============================================
        # FALLBACK 2: OCR (Last resort before VLM)
        # ============================================
        if self.config.extraction.enable_ocr and file_type in [FileType.PDF, FileType.PNG, FileType.JPG, FileType.JPEG]:
            ocr_start = time.time()
            
            if file_type == FileType.PDF:
                ocr_docs = await OCRLoader.load_pdf_with_ocr(file_path, executor)
            else:
                ocr_docs = await OCRLoader.load_image_with_ocr(file_path, executor)
            
            ocr_elapsed = time.time() - ocr_start
            ocr_char_count = sum(len(doc.page_content) for doc in ocr_docs)
            
            logger.info(
                "OCR attempted",
                file_path=str(file_path),
                ocr_char_count=ocr_char_count,
                ocr_time_seconds=f"{ocr_elapsed:.3f}",
                standard_char_count=standard_char_count,
                improvement=f"{ocr_char_count - standard_char_count:+d} chars"
            )
            
            if ocr_char_count > standard_char_count:
                logger.info(
                    "OCR improved extraction",
                    file_path=str(file_path),
                    ocr_chars=ocr_char_count,
                    ocr_time_seconds=f"{ocr_elapsed:.3f}",
                    standard_chars=standard_char_count
                )
                return ocr_docs, ExtractionStrategy.OCR, True, docling_elapsed, ocr_elapsed, 0.0
        
        # ============================================
        # FALLBACK 3: VLM
        # ============================================
        if self.vlm_extractor:
            vlm_start = time.time()
            vlm_docs = await self.vlm_extractor.extract_with_vlm(file_path, executor)
            vlm_elapsed = time.time() - vlm_start
            
            vlm_char_count = sum(len(doc.page_content) for doc in vlm_docs)
            
            # Estimate cost ($0.01 per second for GPT-4 Vision)
            vlm_cost = vlm_elapsed * 0.01
            
            logger.info(
                "VLM attempted",
                file_path=str(file_path),
                vlm_char_count=vlm_char_count,
                vlm_time_seconds=f"{vlm_elapsed:.3f}",
                vlm_cost_estimate=f"${vlm_cost:.3f}",
                standard_char_count=standard_char_count,
                improvement=f"{vlm_char_count - standard_char_count:+d} chars"
            )
            
            if vlm_char_count > standard_char_count:
                logger.info(
                    "VLM improved extraction",
                    file_path=str(file_path),
                    vlm_chars=vlm_char_count,
                    vlm_time_seconds=f"{vlm_elapsed:.3f}",
                    vlm_cost_estimate=f"${vlm_cost:.3f}",
                    standard_chars=standard_char_count
                )
                return vlm_docs, ExtractionStrategy.VLM, True, docling_elapsed, ocr_elapsed, vlm_elapsed
        
        # ============================================
        # All fallbacks failed or didn't improve
        # ============================================
        if standard_docs:
            logger.warning(
                "No fallback improved extraction",
                file_path=str(file_path),
                standard_chars=standard_char_count,
                docling_chars=0 if not docling_elapsed else "not_checked",
                strategy="using_standard_best_effort"
            )
            return standard_docs, ExtractionStrategy.STANDARD, True, docling_elapsed, ocr_elapsed, vlm_elapsed
        
        logger.error(
            "Complete extraction failure",
            file_path=str(file_path),
            all_strategies_failed=True
        )
        return [], ExtractionStrategy.FAILED, True, docling_elapsed, ocr_elapsed, vlm_elapsed