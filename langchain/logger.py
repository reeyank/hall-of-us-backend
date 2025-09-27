"""
Centralized logging module for the LangChain components.

This module provides a configurable logging system that can be enabled/disabled
through the config module. It supports different log levels, formatters, and
output destinations.
"""

import logging
import sys
from typing import Union
from pathlib import Path

from .config import CONFIG


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record):
        # Create the log message format
        log_format = "[{asctime}] [{levelname:8}] [{name}] {message}"

        if self.use_colors and record.levelname in self.COLORS:
            # Add color codes
            levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record.levelname = levelname

        formatter = logging.Formatter(
            log_format, style="{", datefmt="%Y-%m-%d %H:%M:%S"
        )
        return formatter.format(record)


class LoggerManager:
    """Manages logging configuration and provides logger instances"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            LoggerManager._initialized = True

    def _setup_logging(self):
        """Configure the logging system based on config settings"""
        # Get logging configuration
        logging_config = CONFIG.get("logging", {})

        # Determine if logging is enabled
        self.enabled = logging_config.get("enabled", True)

        if not self.enabled:
            # Disable logging completely
            logging.disable(logging.CRITICAL)
            return

        # Configure logging level
        log_level = logging_config.get("level", "INFO").upper()
        level = getattr(logging, log_level, logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Setup console handler
        if logging_config.get("console_output", True):
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(level)
            console_formatter = CustomFormatter(
                use_colors=logging_config.get("use_colors", True)
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # Setup file handler if configured
        log_file = logging_config.get("file_path")
        if log_file:
            self._setup_file_handler(root_logger, log_file, level)

        # Configure specific logger levels
        logger_levels = logging_config.get("logger_levels", {})
        for logger_name, logger_level in logger_levels.items():
            specific_logger = logging.getLogger(logger_name)
            specific_level = getattr(logging, logger_level.upper(), logging.INFO)
            specific_logger.setLevel(specific_level)

    def _setup_file_handler(self, logger: logging.Logger, log_file: str, level: int):
        """Setup file handler for logging"""
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create file handler
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)

            # Use a simpler format for file output (no colors)
            file_format = "[{asctime}] [{levelname:8}] [{name}] {message}"
            file_formatter = logging.Formatter(
                file_format, style="{", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            logger.addHandler(file_handler)

        except (OSError, IOError) as e:
            # If file logging fails, just log to console
            console_logger = logging.getLogger(__name__)
            console_logger.warning(f"Failed to setup file logging: {e}")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name"""
        logger = logging.getLogger(name)

        if not self.enabled:
            # Return a no-op logger if logging is disabled
            return logging.getLogger("disabled")

        return logger

    def is_enabled(self) -> bool:
        """Check if logging is enabled"""
        return self.enabled

    def set_level(self, level: Union[str, int]):
        """Dynamically change the logging level"""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        for handler in root_logger.handlers:
            handler.setLevel(level)


# Global logger manager instance
_logger_manager = LoggerManager()


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__ from the calling module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return _logger_manager.get_logger(name)


def is_logging_enabled() -> bool:
    """Check if logging is enabled"""
    return _logger_manager.is_enabled()


def set_log_level(level: Union[str, int]):
    """
    Dynamically change the logging level.

    Args:
        level: Log level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               or logging level constant
    """
    _logger_manager.set_level(level)


# Convenience function for creating module loggers
def create_module_logger(module_name: str) -> logging.Logger:
    """Create a logger for a specific module"""
    return get_logger(f"langchain.{module_name}")


# Export commonly used logging functions
__all__ = [
    "get_logger",
    "is_logging_enabled",
    "set_log_level",
    "create_module_logger",
    "LoggerManager",
    "CustomFormatter",
]
