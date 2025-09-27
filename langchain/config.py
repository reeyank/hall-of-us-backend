"""
Configuration for LangChain components
"""

import os
from typing import Dict, Any

# API Configuration
DEFAULT_CONFIG = {
    "wrapper": {
        "max_retries": 3,
        "timeout_seconds": 30,
        "enable_logging": True
    },
    "image_tagging": {
        "default_max_tags": 10,
        "default_confidence_threshold": 0.5,
        "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        "max_image_size_mb": 10
    },
    "filter_generation": {
        "default_max_filters": 5,
        "confidence_threshold": 0.7,
        "enable_alternatives": True
    }
}

# Environment variable overrides
def get_config() -> Dict[str, Any]:
    """Get configuration with environment variable overrides"""
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables if they exist
    if os.getenv("LANGCHAIN_MAX_RETRIES"):
        config["wrapper"]["max_retries"] = int(os.getenv("LANGCHAIN_MAX_RETRIES"))
    
    if os.getenv("LANGCHAIN_TIMEOUT"):
        config["wrapper"]["timeout_seconds"] = int(os.getenv("LANGCHAIN_TIMEOUT"))
    
    if os.getenv("IMAGE_MAX_TAGS"):
        config["image_tagging"]["default_max_tags"] = int(os.getenv("IMAGE_MAX_TAGS"))
    
    if os.getenv("FILTER_MAX_COUNT"):
        config["filter_generation"]["default_max_filters"] = int(os.getenv("FILTER_MAX_COUNT"))
    
    return config

# Global config instance
CONFIG = get_config()