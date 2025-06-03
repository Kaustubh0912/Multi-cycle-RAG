import logging
import sys
from typing import Any, Dict, Optional

# Configure basic logger format for structured logging
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create base logger
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)


class StructuredLogger:
    """Structured logger with context support"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def with_context(self, **kwargs) -> "StructuredLogger":
        """Create a logger with additional context"""
        logger_copy = StructuredLogger(self.logger.name)
        logger_copy.context = {**self.context, **kwargs}
        return logger_copy

    def _format_message(
        self, message: str, extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format message with context and extra data"""
        if not extra and not self.context:
            return message

        context_data = {**self.context}
        if extra:
            context_data.update(extra)

        context_str = " | ".join(f"{k}={v}" for k, v in context_data.items())
        return f"{message} | {context_str}" if context_str else message

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data"""
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data"""
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data"""
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data"""
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data"""
        self.logger.critical(self._format_message(message, kwargs))

    def exception(self, message: str, **kwargs) -> None:
        """Log exception message with structured data and traceback"""
        self.logger.exception(self._format_message(message, kwargs))


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name)


# Default logger instance
logger = get_logger("rag")


# Configure specific module loggers
def configure_loggers():
    """Configure module-specific loggers"""
    # Set third-party loggers to higher log levels to reduce noise
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Configure on import
configure_loggers()
