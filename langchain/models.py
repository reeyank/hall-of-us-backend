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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Successful Response",
                    "description": "Example of a successful API response",
                    "value": {
                        "success": True,
                        "data": {"result": "Image tagged successfully", "tags": 5},
                        "error": None,
                        "timestamp": "2025-09-27T10:30:00Z",
                        "execution_time_ms": 1250.5,
                    },
                },
                {
                    "title": "Error Response - API Key Missing",
                    "description": "Example of failed response due to missing API key",
                    "value": {
                        "success": False,
                        "data": None,
                        "error": "OpenAI API key not provided. Set OPENAI_API_KEY environment variable.",
                        "timestamp": "2025-09-27T10:30:00Z",
                        "execution_time_ms": 50.0,
                    },
                },
                {
                    "title": "Error Response - Rate Limited",
                    "description": "Example of failed response due to rate limiting",
                    "value": {
                        "success": False,
                        "data": None,
                        "error": "Rate limit exceeded. Please try again in 60 seconds.",
                        "timestamp": "2025-09-27T10:30:00Z",
                        "execution_time_ms": 2000.0,
                    },
                },
                {
                    "title": "Error Response - Invalid Input",
                    "description": "Example of failed response due to invalid input",
                    "value": {
                        "success": False,
                        "data": None,
                        "error": "Neither image_url nor image_base64 provided. At least one is required.",
                        "timestamp": "2025-09-27T10:30:00Z",
                        "execution_time_ms": 25.0,
                    },
                },
            ]
        }
    }


class ImageTaggingRequest(BaseModel):
    """Request model for image tagging"""

    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    max_tags: int = 10
    confidence_threshold: float = 0.5
    categories: Optional[List[str]] = None  # Specific categories to focus on

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Valid URL Request",
                    "description": "Standard image tagging request with URL",
                    "value": {
                        "image_url": "https://example.com/image.jpg",
                        "max_tags": 5,
                        "confidence_threshold": 0.8,
                        "categories": ["animals", "nature"],
                    },
                },
                {
                    "title": "Valid Base64 Request",
                    "description": "Image tagging request with base64 encoded image",
                    "value": {
                        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                        "max_tags": 10,
                        "confidence_threshold": 0.6,
                    },
                },
                {
                    "title": "Invalid - No Image Source",
                    "description": "Failure example: no image source provided",
                    "value": {"max_tags": 5, "confidence_threshold": 0.7},
                },
                {
                    "title": "Invalid - Bad URL",
                    "description": "Failure example: malformed URL",
                    "value": {
                        "image_url": "not-a-valid-url",
                        "max_tags": 5,
                        "confidence_threshold": 0.7,
                    },
                },
                {
                    "title": "Invalid - Bad Parameters",
                    "description": "Failure example: invalid parameter values",
                    "value": {
                        "image_url": "https://example.com/image.jpg",
                        "max_tags": -1,
                        "confidence_threshold": 1.5,
                    },
                },
            ]
        }
    }


class ImageTaggingResponse(BaseModel):
    """Response model for image tagging"""

    tags: List[
        Dict[str, Any]
    ]  # [{"tag": "cat", "confidence": 0.95, "category": "animal"}, ...]
    image_metadata: Optional[Dict[str, Any]] = None
    processing_info: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Successful Tagging",
                    "description": "Example of successful image tagging response",
                    "value": {
                        "tags": [
                            {"tag": "dog", "confidence": 0.95, "category": "animal"},
                            {
                                "tag": "outdoor",
                                "confidence": 0.87,
                                "category": "environment",
                            },
                            {"tag": "grass", "confidence": 0.82, "category": "nature"},
                        ],
                        "image_metadata": {
                            "width": 1920,
                            "height": 1080,
                            "format": "JPEG",
                            "size_bytes": 245760,
                        },
                        "processing_info": {
                            "model_used": "gpt-4o",
                            "processing_time_ms": 2150,
                            "api_version": "2024-02-01",
                        },
                    },
                },
                {
                    "title": "Empty Tags - Low Confidence",
                    "description": "Response when no tags meet confidence threshold",
                    "value": {
                        "tags": [],
                        "image_metadata": {
                            "width": 100,
                            "height": 100,
                            "format": "PNG",
                            "size_bytes": 1024,
                        },
                        "processing_info": {
                            "model_used": "gpt-4o",
                            "processing_time_ms": 1850,
                            "reason": "No tags found above confidence threshold of 0.8",
                        },
                    },
                },
                {
                    "title": "Partial Failure",
                    "description": "Response with some tags but processing warnings",
                    "value": {
                        "tags": [
                            {
                                "tag": "unclear",
                                "confidence": 0.45,
                                "category": "general",
                            }
                        ],
                        "image_metadata": {
                            "width": 50,
                            "height": 50,
                            "format": "GIF",
                            "size_bytes": 512,
                        },
                        "processing_info": {
                            "model_used": "gpt-4o",
                            "processing_time_ms": 3200,
                            "warnings": [
                                "Image resolution too low",
                                "Poor image quality detected",
                            ],
                        },
                    },
                },
            ]
        }
    }


class FilterGenerationRequest(BaseModel):
    """Request model for filter generation"""

    natural_language_query: str
    available_filters: List[Dict[str, Any]]  # List of available filter definitions
    context: Optional[Dict[str, Any]] = (
        None  # Additional context about the data being filtered
    )
    max_filters: int = 5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Valid Filter Request",
                    "description": "Standard filter generation request",
                    "value": {
                        "natural_language_query": "Show me all users who are active and from California",
                        "available_filters": [
                            {
                                "name": "status",
                                "type": "enum",
                                "values": ["active", "inactive"],
                            },
                            {"name": "location", "type": "string", "searchable": True},
                            {"name": "age", "type": "number", "min": 0, "max": 120},
                        ],
                        "context": {"data_type": "users", "total_records": 10000},
                        "max_filters": 3,
                    },
                },
                {
                    "title": "Complex Query Request",
                    "description": "Filter request with complex natural language",
                    "value": {
                        "natural_language_query": "Find products that are on sale, cost less than $100, and have good ratings",
                        "available_filters": [
                            {"name": "on_sale", "type": "boolean"},
                            {"name": "price", "type": "number", "min": 0},
                            {"name": "rating", "type": "number", "min": 1, "max": 5},
                        ],
                        "max_filters": 5,
                    },
                },
                {
                    "title": "Invalid - Empty Query",
                    "description": "Failure example: empty natural language query",
                    "value": {
                        "natural_language_query": "",
                        "available_filters": [
                            {
                                "name": "status",
                                "type": "enum",
                                "values": ["active", "inactive"],
                            }
                        ],
                        "max_filters": 3,
                    },
                },
                {
                    "title": "Invalid - No Available Filters",
                    "description": "Failure example: empty available filters list",
                    "value": {
                        "natural_language_query": "Show me active users",
                        "available_filters": [],
                        "max_filters": 3,
                    },
                },
                {
                    "title": "Invalid - Bad Parameters",
                    "description": "Failure example: invalid max_filters value",
                    "value": {
                        "natural_language_query": "Show me active users",
                        "available_filters": [
                            {
                                "name": "status",
                                "type": "enum",
                                "values": ["active", "inactive"],
                            }
                        ],
                        "max_filters": 0,
                    },
                },
            ]
        }
    }


class FilterGenerationResponse(BaseModel):
    """Response model for filter generation"""

    generated_filters: List[Dict[str, Any]]
    explanation: str
    confidence_score: float
    alternative_suggestions: Optional[List[Dict[str, Any]]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Successful Filter Generation",
                    "description": "Example of successful filter generation response",
                    "value": {
                        "generated_filters": [
                            {
                                "field": "status",
                                "operator": "equals",
                                "value": "active",
                                "confidence": 0.95,
                            },
                            {
                                "field": "location",
                                "operator": "contains",
                                "value": "California",
                                "confidence": 0.87,
                            },
                        ],
                        "explanation": "Based on your query 'active users from California', I created filters for status=active and location containing California.",
                        "confidence_score": 0.91,
                        "alternative_suggestions": [
                            {
                                "field": "state",
                                "operator": "equals",
                                "value": "CA",
                                "reason": "Alternative state code format",
                            }
                        ],
                    },
                },
                {
                    "title": "Partial Match Response",
                    "description": "Response when only some filters could be generated",
                    "value": {
                        "generated_filters": [
                            {
                                "field": "price",
                                "operator": "less_than",
                                "value": 100,
                                "confidence": 0.75,
                            }
                        ],
                        "explanation": "I could identify a price filter, but 'good ratings' was too vague to create a specific filter.",
                        "confidence_score": 0.65,
                        "alternative_suggestions": [
                            {
                                "field": "rating",
                                "operator": "greater_than",
                                "value": 4.0,
                                "reason": "Assumption: 'good' means rating > 4.0",
                            }
                        ],
                    },
                },
                {
                    "title": "Low Confidence Response",
                    "description": "Response when query is unclear or ambiguous",
                    "value": {
                        "generated_filters": [],
                        "explanation": "The query was too ambiguous to generate specific filters. Please provide more specific criteria.",
                        "confidence_score": 0.25,
                        "alternative_suggestions": [
                            {
                                "suggestion": "Try specifying exact field names or values",
                                "example": "Instead of 'good products', try 'products with rating > 4'",
                            }
                        ],
                    },
                },
            ]
        }
    }
