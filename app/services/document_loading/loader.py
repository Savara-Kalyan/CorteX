import asyncio
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredPDFLoader,
)

from .models import FileType
from .exceptions import ExtractionException
from ...logging import get_logger


logger = get_logger(__name__)


class FileTypeDetector:
    """
    Detect file type from extension.
    """
    
    FILE_TYPE_MAP = {
        ".pdf": FileType.PDF,
        ".md": FileType.MARKDOWN,
        ".docx": FileType.DOCX,
        ".txt": FileType.TXT,
        ".png": FileType.PNG,
        ".jpg": FileType.JPG,
        ".jpeg": FileType.JPEG,
    }
    
    @classmethod
    def detect(cls, file_path: Path) -> FileType:
        """
        Detect file type.
        """

        suffix = file_path.suffix.lower()
        return cls.FILE_TYPE_MAP.get(suffix, FileType.UNKNOWN)
    
    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """
        Check if supported.
        """

        return cls.detect(file_path) != FileType.UNKNOWN


class StandardLoader:
    """
    Standard extraction using PDFPlumber, etc.
    """
    
    @staticmethod
    async def load_pdf(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract from PDF using PDFPlumber.
        
        Fast for text PDFs, but fails on scanned PDFs.
        """
        loop = asyncio.get_event_loop()
        try:
            loader = PDFPlumberLoader(str(file_path))
            docs = await loop.run_in_executor(executor, loader.load)
            logger.debug("PDF loaded", file_path=str(file_path), doc_count=len(docs))
            return docs
        except Exception as e:
            logger.warning("PDF load failed", file_path=str(file_path), error=str(e))
            return []
    
    @staticmethod
    async def load_markdown(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract from Markdown.
        """

        loop = asyncio.get_event_loop()
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            docs = await loop.run_in_executor(executor, loader.load)
            logger.debug("Markdown loaded", file_path=str(file_path), doc_count=len(docs))
            return docs
        except Exception as e:
            logger.warning("Markdown load failed", file_path=str(file_path), error=str(e))
            return []
    
    @staticmethod
    async def load_docx(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract from DOCX.
        """

        loop = asyncio.get_event_loop()
        try:
            loader = Docx2txtLoader(str(file_path))
            docs = await loop.run_in_executor(executor, loader.load)
            logger.debug("DOCX loaded", file_path=str(file_path), doc_count=len(docs))
            return docs
        except Exception as e:
            logger.warning("DOCX load failed", file_path=str(file_path), error=str(e))
            return []
    
    @staticmethod
    async def load_text(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract from TXT.
        """

        loop = asyncio.get_event_loop()
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = await loop.run_in_executor(executor, loader.load)
            logger.debug("TXT loaded", file_path=str(file_path), doc_count=len(docs))
            return docs
        except Exception as e:
            logger.warning("TXT load failed", file_path=str(file_path), error=str(e))
            return []
    
    @classmethod
    async def extract(
        cls,
        file_path: Path,
        file_type: FileType,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract using standard loader.
        """
        
        try:
            if file_type == FileType.PDF:
                return await cls.load_pdf(file_path, executor)
            elif file_type == FileType.MARKDOWN:
                return await cls.load_markdown(file_path, executor)
            elif file_type == FileType.DOCX:
                return await cls.load_docx(file_path, executor)
            elif file_type == FileType.TXT:
                return await cls.load_text(file_path, executor)
            else:
                logger.warning("No loader for file type", file_path=str(file_path), file_type=file_type.value)
                return []
        
        except Exception as e:
            raise ExtractionException(
                f"Standard extraction failed: {str(e)}",
                file_path=str(file_path),
            )


class DoclingLoader:
    """
    Smart PDF extraction using Docling.
    
    Docling is better than OCR for most PDFs:
    - Handles both text and scanned PDFs
    - Preserves layout (tables, figures)
    - Fast (< 1s per PDF)
    - Handles complex documents
    
    This is the NEW primary fallback (before OCR).
    """
    
    @staticmethod
    async def load_pdf_with_docling(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """Extract PDF using Docling.
        
        Async wrapper around blocking Docling call.
        """
        loop = asyncio.get_event_loop()
        try:
            from docling.document_converter import DocumentConverter
            
            def _extract_with_docling():
                """Blocking Docling extraction."""
                converter = DocumentConverter()
                result = converter.convert(str(file_path))
                
                # Export to text
                all_text = result.document.export_to_text()
                
                # Create Document
                doc = Document(
                    page_content=all_text,
                    metadata={
                        "source": str(file_path),
                        "extraction_method": "docling",
                        "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                    }
                )
                return [doc]
            
            # Run in thread pool (blocking I/O)
            docs = await loop.run_in_executor(executor, _extract_with_docling)
            
            char_count = sum(len(doc.page_content) for doc in docs)
            logger.debug(
                "Docling extracted",
                file_path=str(file_path),
                char_count=char_count,
                doc_count=len(docs),
                extraction_method="docling"
            )
            return docs
        
        except ImportError:
            logger.warning(
                "Docling not installed",
                file_path=str(file_path),
                suggestion="Install with: pip install docling"
            )
            return []
        
        except Exception as e:
            logger.warning(
                "Docling extraction failed",
                file_path=str(file_path),
                error=str(e)
            )
            return []


class OCRLoader:
    """
    OCR-based extraction (Last resort fallback).
    
    Uses Tesseract for text recognition.
    Slower and less accurate than Docling, so use as last resort.
    """
    
    @staticmethod
    async def load_pdf_with_ocr(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """Extract from PDF using OCR."""
        loop = asyncio.get_event_loop()
        try:
            loader = UnstructuredPDFLoader(str(file_path))
            docs = await loop.run_in_executor(executor, loader.load)
            logger.info(
                "OCR PDF loaded",
                file_path=str(file_path),
                doc_count=len(docs),
                strategy="ocr"
            )
            return docs
        except Exception as e:
            logger.warning(
                "OCR PDF load failed",
                file_path=str(file_path),
                error=str(e)
            )
            return []
    
    @staticmethod
    async def load_image_with_ocr(
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """Extract from image using OCR."""
        loop = asyncio.get_event_loop()
        try:
            loader = UnstructuredImageLoader(str(file_path))
            docs = await loop.run_in_executor(executor, loader.load)
            logger.info(
                "OCR image loaded",
                file_path=str(file_path),
                doc_count=len(docs),
                strategy="ocr"
            )
            return docs
        except Exception as e:
            logger.warning(
                "OCR image load failed",
                file_path=str(file_path),
                error=str(e)
            )
            return []


class VLMLoader:
    """
    Vision Language Model extraction (Most expensive).
    
    Uses GPT-4 Vision as absolute last resort.
    Very smart but costs money ($$$).
    """
    
    def __init__(self, model: str = "gpt-4-vision-preview", api_key: Optional[str] = None):
        """Initialize VLM."""
        try:
            from langchain_openai import ChatOpenAI
            self.model = model
            self.llm = ChatOpenAI(model=model, api_key=api_key, max_tokens=2000)
        except ImportError:
            raise ExtractionException("langchain-openai not installed")
    
    async def extract_with_vlm(
        self,
        file_path: Path,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[Document]:
        """
        Extract using GPT-4 Vision.
        """
        
        loop = asyncio.get_event_loop()
        
        try:
            from langchain_core.messages import HumanMessage
            import base64
            
            def _encode_and_send():
                """Blocking VLM call."""
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                base64_data = base64.b64encode(file_data).decode("utf-8")
                
                file_type = FileTypeDetector.detect(file_path)
                media_type_map = {
                    FileType.PDF: "application/pdf",
                    FileType.PNG: "image/png",
                    FileType.JPG: "image/jpeg",
                    FileType.JPEG: "image/jpeg",
                }
                media_type = media_type_map.get(file_type, "application/octet-stream")
                
                message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract and summarize all text content from this document. Return only the extracted text.",
                        },
                    ]
                )
                
                response = self.llm.invoke([message])
                return response.content
            
            extracted_text = await loop.run_in_executor(executor, _encode_and_send)
            
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": str(file_path),
                    "extraction_method": "vlm",
                    "model": self.model,
                }
            )
            
            logger.info(
                "VLM extracted",
                file_path=str(file_path),
                char_count=len(extracted_text),
                strategy="vlm",
                model=self.model
            )
            return [doc]
        
        except Exception as e:
            logger.error(
                "VLM extraction failed",
                file_path=str(file_path),
                error=str(e)
            )
            return []