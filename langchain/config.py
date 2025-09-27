"""
Configuration for LangChain components
"""

import os
from typing import Dict, Any

# API Configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "wrapper": {"max_retries": 3, "timeout_seconds": 30, "enable_logging": True},
    "image_tagging": {
        "default_max_tags": 10,
        "default_confidence_threshold": 0.5,
        "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        "max_image_size_mb": 10,
    },
    "filter_generation": {
        "default_max_filters": 5,
        "confidence_threshold": 0.7,
        "enable_alternatives": True,
    },
    "logging": {
        "enabled": True,
        "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "console_output": True,
        "use_colors": True,
        "file_path": "logs/langchain.log",  # Set to None to disable file logging
        "logger_levels": {
            # Specific logger levels can be set here
            "langchain.wrapper": "INFO",
            "langchain.image_tagging": "INFO",
            "langchain.filter_generation": "INFO",
            # Suppress noisy third-party loggers
            "openai": "WARNING",
            "httpcore": "WARNING",
            "httpx": "WARNING",
        },
    },
}


# Environment variable overrides
def get_config() -> Dict[str, Any]:
    """Get configuration with environment variable overrides"""
    import copy

    config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

    # Override with environment variables if they exist
    langchain_retries = os.getenv("LANGCHAIN_MAX_RETRIES")
    if langchain_retries:
        config["wrapper"]["max_retries"] = int(langchain_retries)

    langchain_timeout = os.getenv("LANGCHAIN_TIMEOUT")
    if langchain_timeout:
        config["wrapper"]["timeout_seconds"] = int(langchain_timeout)

    image_max_tags = os.getenv("IMAGE_MAX_TAGS")
    if image_max_tags:
        config["image_tagging"]["default_max_tags"] = int(image_max_tags)

    filter_max_count = os.getenv("FILTER_MAX_COUNT")
    if filter_max_count:
        config["filter_generation"]["default_max_filters"] = int(filter_max_count)

    # Logging environment variable overrides
    logging_enabled = os.getenv("LANGCHAIN_LOGGING_ENABLED")
    if logging_enabled:
        config["logging"]["enabled"] = logging_enabled.lower() == "true"

    log_level = os.getenv("LANGCHAIN_LOG_LEVEL")
    if log_level:
        config["logging"]["level"] = log_level.upper()

    log_file = os.getenv("LANGCHAIN_LOG_FILE")
    if log_file:
        config["logging"]["file_path"] = log_file

    log_colors = os.getenv("LANGCHAIN_LOG_COLORS")
    if log_colors:
        config["logging"]["use_colors"] = log_colors.lower() == "true"

    return config


# Global config instance
CONFIG = get_config()
