from typing import List, Tuple
from langchain_core.documents import Document

from ...logging import get_logger


logger = get_logger(__name__)


class ContentQualityAnalyzer:
    """
    Analyze extraction quality.
    """
    
    def __init__(self, char_count_threshold: int = 500):
        """
        Initialize.
        """

        self.char_count_threshold = char_count_threshold
    
    def analyze(self, documents: List[Document]) -> Tuple[int, bool]:
        """
        Analyze quality.
        
        Returns:
            (char_count, should_fallback)
        """
        
        if not documents:
            return 0, True
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        should_fallback = total_chars < self.char_count_threshold
        
        logger.debug(
            "Quality analysis",
            char_count=total_chars,
            threshold=self.char_count_threshold,
            should_fallback=should_fallback
        )
        
        return total_chars, should_fallback