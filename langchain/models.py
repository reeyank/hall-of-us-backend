"""
Pydantic models for the LangChain wrapper
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

# Import the new logger module
from .logger import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class APIResponse(BaseModel):
    """Standardized API response format"""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime
    execution_time_ms: Optional[float] = None


class ImageTaggingRequest(BaseModel):
    """Request model for image tagging"""

    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    max_tags: int = 10
    confidence_threshold: float = 0.5
    categories: Optional[List[str]] = None  # Specific categories to focus on


class ImageTaggingResponse(BaseModel):
    """Response model for image tagging"""

    tags: List[
        Dict[str, Any]
    ]  # [{"tag": "cat", "confidence": 0.95, "category": "animal"}, ...]
    image_metadata: Optional[Dict[str, Any]] = None
    processing_info: Optional[Dict[str, Any]] = None


class FilterGenerationRequest(BaseModel):
    """Request model for filter generation"""

    natural_language_query: str
    available_filters: List[Dict[str, Any]]  # List of available filter definitions
    context: Optional[Dict[str, Any]] = (
        None  # Additional context about the data being filtered
    )
    max_filters: int = 5


class FilterGenerationResponse(BaseModel):
    """Response model for filter generation"""

    generated_filters: List[Dict[str, Any]]
    explanation: str
    confidence_score: float
    alternative_suggestions: Optional[List[Dict[str, Any]]] = None
