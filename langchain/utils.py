"""
Utility functions for LangChain components
"""

import base64
import hashlib
import mimetypes
from typing import Dict, Any, Optional
from datetime import datetime

from .config import CONFIG


def validate_image_format(filename: str) -> bool:
    """Validate if the image format is supported"""
    if not filename:
        return False

    ext = filename.lower().split('.')[-1]
    return ext in CONFIG["image_tagging"]["supported_formats"]


def validate_image_size(image_data: bytes) -> bool:
    """Validate if the image size is within limits"""
    size_mb = len(image_data) / (1024 * 1024)
    return size_mb <= CONFIG["image_tagging"]["max_image_size_mb"]


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image file to base64"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            if validate_image_size(image_data):
                return base64.b64encode(image_data).decode('utf-8')
            else:
                raise ValueError(f"Image size exceeds {CONFIG['image_tagging']['max_image_size_mb']}MB limit")
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes"""
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def get_image_info(image_data: bytes, source: str = "unknown") -> Dict[str, Any]:
    """Get metadata information about an image"""
    return {
        "size_bytes": len(image_data),
        "size_mb": round(len(image_data) / (1024 * 1024), 2),
        "source": source,
        "hash": hashlib.md5(image_data).hexdigest(),
        "processed_at": datetime.now().isoformat()
    }


def sanitize_filter_query(query: str) -> str:
    """Sanitize natural language query for filter generation"""
    # Remove potentially harmful characters
    sanitized = query.strip()
    sanitized = ''.join(char for char in sanitized if char.isprintable())
    return sanitized[:500]  # Limit length


def validate_filter_structure(filter_config: Dict[str, Any]) -> bool:
    """Validate that a filter configuration has the required structure"""
    required_fields = ["type", "field", "operator"]
    return all(field in filter_config for field in required_fields)


def format_execution_time(execution_time_ms: Optional[float]) -> str:
    """Format execution time for display"""
    if execution_time_ms is None:
        return "N/A"

    if execution_time_ms < 1000:
        return f"{execution_time_ms:.2f}ms"
    else:
        return f"{execution_time_ms/1000:.2f}s"


def create_error_response(error_message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "success": False,
        "error": error_message,
        "error_code": error_code,
        "timestamp": datetime.now().isoformat()
    }
