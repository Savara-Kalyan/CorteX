import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass


class LogLevel(Enum):
    """
    Log levels
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LoggingConfig:
    """
    logging config.
    """
    
    level: LogLevel = LogLevel.INFO
    log_file: str = "logs/cortex.log"
    console_enabled: bool = True
    file_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 3
    
    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """
        Load from environment variables.
        """

        return cls(
            level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "logs/cortex.log"),
            console_enabled=os.getenv("LOG_CONSOLE", "true").lower() == "true",
            file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
        )


def get_logging_config() -> LoggingConfig:
    """
    Get config from environment.
    """
    
    return LoggingConfig.from_env()