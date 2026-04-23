import os
import sys
import json
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LoggingConfig:
    level: LogLevel = LogLevel.INFO
    log_file: str = "logs/cortex.log"
    console_enabled: bool = True
    file_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 3

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        return cls(
            level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "logs/cortex.log"),
            console_enabled=os.getenv("LOG_CONSOLE", "true").lower() == "true",
            file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
        )


def get_logging_config() -> LoggingConfig:
    return LoggingConfig.from_env()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
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
    def format(self, record: logging.LogRecord) -> str:
        return f"[{record.levelname}] {record.getMessage()}"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self._context: Dict[str, Any] = {}

        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.level.value)
        self.logger.handlers = []

        if config.console_enabled:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(config.level.value)
            handler.setFormatter(ConsoleFormatter())
            self.logger.addHandler(handler)

        if config.file_enabled:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                filename=config.log_file,
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count,
            )
            handler.setLevel(config.level.value)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def set_context(self, **kwargs):
        self._context.update(kwargs)

    def clear_context(self):
        self._context.clear()

    def _log(self, level: str, message: str, **kwargs):
        extra_data = {**self._context, **kwargs}
        record = self.logger.makeRecord(
            name=self.logger.name, level=getattr(logging, level),
            fn=self.logger.name, lno=0, msg=message, args=(), exc_info=None,
        )
        if extra_data:
            record.extra_data = extra_data
        self.logger.handle(record)

    def debug(self, message: str, **kwargs): self._log("DEBUG", message, **kwargs)
    def info(self, message: str, **kwargs): self._log("INFO", message, **kwargs)
    def warning(self, message: str, **kwargs): self._log("WARNING", message, **kwargs)
    def error(self, message: str, **kwargs): self._log("ERROR", message, **kwargs)
    def critical(self, message: str, **kwargs): self._log("CRITICAL", message, **kwargs)


_config: Optional[LoggingConfig] = None
_loggers: Dict[str, Logger] = {}


def configure(config: LoggingConfig):
    global _config
    _config = config


def get_logger(name: str) -> Logger:
    global _config
    if _config is None:
        _config = get_logging_config()
    if name not in _loggers:
        _loggers[name] = Logger(name, _config)
    return _loggers[name]
