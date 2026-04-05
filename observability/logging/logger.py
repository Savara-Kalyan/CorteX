import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

from .config import LoggingConfig, get_logging_config


class JSONFormatter(logging.Formatter):
    """
    Format logs as JSON.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Convert log record to JSON.
        """
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
 
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """
    Pretty console format.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Pretty format for terminal.
        """

        return f"[{record.levelname}] {record.getMessage()}"


class Logger:
    """
    Logger with JSON and console output.
    """
    
    def __init__(self, name: str, config: LoggingConfig):
        """
        Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
            config: Logging configuration
        """
        
        self.name = name
        self.config = config
        self._context: Dict[str, Any] = {}
        
        # Create Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.level.value)
        self.logger.handlers = []
        
        # Add console handler
        if config.console_enabled:
            self._add_console_handler(config)
        
        # Add file handler
        if config.file_enabled:
            self._add_file_handler(config)
    
    def _add_console_handler(self, config: LoggingConfig):
        """
        Add console handler for terminal output.
        """
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(config.level.value)
        handler.setFormatter(ConsoleFormatter())
        self.logger.addHandler(handler)
    
    def _add_file_handler(self, config: LoggingConfig):
        """
        Add file handler for JSON logs.
        """
        
        # Create logs directory
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (auto-rotates when size exceeded)
        handler = RotatingFileHandler(
            filename=config.log_file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
        )
        handler.setLevel(config.level.value)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def set_context(self, **kwargs):
        """
        Set context for subsequent logs.
        
        Example:
            logger.set_context(request_id="req-123", user_id="user-456")
            logger.info("Processing")  # Will include request_id and user_id
        """

        self._context.update(kwargs)
    
    def clear_context(self):
        """
        Clear all context.
        """

        self._context.clear()
    
    def _log(self, level: str, message: str, **kwargs):
        """
        Internal logging method.
        """
        
        # Combine stored context with current kwargs
        extra_data = {**self._context, **kwargs}
        
        # Create log record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=getattr(logging, level),
            fn=self.logger.name,
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        
        # Attach extra data
        if extra_data:
            record.extra_data = extra_data
        
        # Log it
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """
        Debug message.
        """

        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """
        Info message.
        """

        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """
        Warning message.
        """

        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """
        Error message.
        """

        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """
        Critical message.
        """

        self._log("CRITICAL", message, **kwargs)


_config: Optional[LoggingConfig] = None
_loggers: Dict[str, Logger] = {}


def configure(config: LoggingConfig):
    """
    Initialize logging system once at startup.
    
    Example:
        from cortex.core.observability.logging import configure, get_logger
        from cortex.core.observability.logging.config import get_logging_config
        
        config = get_logging_config()
        configure(config)
        
        logger = get_logger(__name__)
    """
    global _config
    _config = config


def get_logger(name: str) -> Logger:
    """
    Get a logger instance.
    
    Args:
        name: Usually __name__
        
    Returns:
        Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Hello world")
    """
    
    global _config
    
    # Initialize if not already done
    if _config is None:
        _config = get_logging_config()
    
    # Create or retrieve logger
    if name not in _loggers:
        _loggers[name] = Logger(name, _config)
    
    return _loggers[name]