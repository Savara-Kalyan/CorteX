
class DocumentLoadingException(Exception):
    """Base exception."""
    
    def __init__(self, message: str, error_code: str, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self):
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class FileDiscoveryException(DocumentLoadingException):
    """File discovery failed."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "FILE_DISCOVERY_ERROR", details)


class ExtractionException(DocumentLoadingException):
    """Extraction failed."""
    
    def __init__(self, message: str, file_path: str = None, details: dict = None):
        if details is None:
            details = {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, "EXTRACTION_ERROR", details)


class ValidationException(DocumentLoadingException):
    """Validation failed."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "VALIDATION_ERROR", details)