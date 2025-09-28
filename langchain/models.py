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


# Utility functions for handling failure cases across all models
class FailureCaseHandler:
    """Centralized handler for common failure cases across all models"""

    @staticmethod
    def log_validation_failure(
        model_name: str, field_name: str, error_msg: str
    ) -> None:
        """Log validation failures consistently"""
        logger.error(f"{model_name}.{field_name}: {error_msg}")

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format"""
        return bool(url and url.startswith(("http://", "https://")))

    @staticmethod
    def is_valid_base64(base64_str: str) -> bool:
        """Validate base64 string format"""
        if not base64_str:
            return False
        try:
            import base64

            base64.b64decode(base64_str, validate=True)
            return True
        except Exception:
            return False


class APIResponse(BaseModel):
    """Standardized API response format"""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime
    execution_time_ms: Optional[float] = None

    @classmethod
    def create_api_key_missing_failure(
        cls, timestamp: datetime, execution_time_ms: float = 50.0
    ) -> "APIResponse":
        """Create failure response for missing API key case"""
        logger.error("API key missing failure case triggered")
        return cls(
            success=False,
            data=None,
            error="OpenAI API key not provided. Set OPENAI_API_KEY environment variable.",
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def create_rate_limited_failure(
        cls, timestamp: datetime, execution_time_ms: float = 2000.0
    ) -> "APIResponse":
        """Create failure response for rate limiting case"""
        logger.warning("Rate limit exceeded failure case triggered")
        return cls(
            success=False,
            data=None,
            error="Rate limit exceeded. Please try again in 60 seconds.",
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def create_invalid_input_failure(
        cls, timestamp: datetime, execution_time_ms: float = 25.0
    ) -> "APIResponse":
        """Create failure response for invalid input case"""
        logger.error("Invalid input failure case triggered")
        return cls(
            success=False,
            data=None,
            error="Neither image_url nor image_base64 provided. At least one is required.",
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
        )

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

    def validate_no_image_source_failure(self) -> bool:
        """Check for no image source failure case"""
        has_source = any([self.image_url, self.image_base64, self.image_path])
        if not has_source:
            logger.error(
                "No image source failure case: no image_url, image_base64, or image_path provided"
            )
            return False
        return True

    def validate_bad_url_failure(self) -> bool:
        """Check for bad URL failure case"""
        if self.image_url and not FailureCaseHandler.is_valid_url(self.image_url):
            logger.error(f"Bad URL failure case: malformed URL '{self.image_url}'")
            return False
        return True

    def validate_bad_parameters_failure(self) -> bool:
        """Check for bad parameters failure case"""
        if self.max_tags < 0:
            logger.error(
                f"Bad parameters failure case: max_tags cannot be negative ({self.max_tags})"
            )
            return False
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            logger.error(
                f"Bad parameters failure case: confidence_threshold must be between 0.0 and 1.0 ({self.confidence_threshold})"
            )
            return False
        return True

    def validate_all_failure_cases(self) -> bool:
        """Validate all documented failure cases"""
        return (
            self.validate_no_image_source_failure()
            and self.validate_bad_url_failure()
            and self.validate_bad_parameters_failure()
        )

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

    @classmethod
    def create_empty_tags_low_confidence_response(
        cls, confidence_threshold: float, metadata: Optional[Dict[str, Any]] = None
    ) -> "ImageTaggingResponse":
        """Create response for empty tags due to low confidence failure case"""
        logger.warning(
            f"Empty tags low confidence case: no tags above threshold {confidence_threshold}"
        )
        return cls(
            tags=[],
            image_metadata=metadata
            or {"width": 100, "height": 100, "format": "PNG", "size_bytes": 1024},
            processing_info={
                "model_used": "gpt-4o",
                "processing_time_ms": 1850,
                "reason": f"No tags found above confidence threshold of {confidence_threshold}",
            },
        )

    @classmethod
    def create_partial_failure_response(
        cls, low_confidence_tags: List[Dict[str, Any]], warnings: List[str]
    ) -> "ImageTaggingResponse":
        """Create response for partial failure case with warnings"""
        logger.warning(f"Partial failure case with warnings: {warnings}")
        return cls(
            tags=low_confidence_tags,
            image_metadata={
                "width": 50,
                "height": 50,
                "format": "GIF",
                "size_bytes": 512,
            },
            processing_info={
                "model_used": "gpt-4o",
                "processing_time_ms": 3200,
                "warnings": warnings,
            },
        )

    def has_quality_issues(self) -> bool:
        """Check if response indicates quality issues"""
        if self.processing_info and "warnings" in self.processing_info:
            quality_warnings = [
                "Image resolution too low",
                "Poor image quality detected",
            ]
            return any(
                warning in self.processing_info["warnings"]
                for warning in quality_warnings
            )
        return False

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

    def validate_empty_query_failure(self) -> bool:
        """Check for empty query failure case"""
        if not self.natural_language_query or self.natural_language_query.strip() == "":
            logger.error("Empty query failure case: natural_language_query is empty")
            return False
        return True

    def validate_no_available_filters_failure(self) -> bool:
        """Check for no available filters failure case"""
        if not self.available_filters or len(self.available_filters) == 0:
            logger.error(
                "No available filters failure case: available_filters list is empty"
            )
            return False
        return True

    def validate_bad_parameters_failure(self) -> bool:
        """Check for bad parameters failure case"""
        if self.max_filters <= 0:
            logger.error(
                f"Bad parameters failure case: max_filters must be positive ({self.max_filters})"
            )
            return False
        return True

    def validate_all_failure_cases(self) -> bool:
        """Validate all documented failure cases"""
        return (
            self.validate_empty_query_failure()
            and self.validate_no_available_filters_failure()
            and self.validate_bad_parameters_failure()
        )

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

    @classmethod
    def create_partial_match_response(
        cls,
        partial_filters: List[Dict[str, Any]],
        reason: str,
        alternatives: Optional[List[Dict[str, Any]]] = None,
    ) -> "FilterGenerationResponse":
        """Create response for partial match failure case"""
        logger.warning(f"Partial match case: {reason}")
        return cls(
            generated_filters=partial_filters,
            explanation=f"I could identify some filters, but {reason}",
            confidence_score=0.65,
            alternative_suggestions=alternatives or [],
        )

    @classmethod
    def create_low_confidence_response(
        cls, reason: str = "query was too ambiguous"
    ) -> "FilterGenerationResponse":
        """Create response for low confidence failure case"""
        logger.warning(f"Low confidence case: {reason}")
        return cls(
            generated_filters=[],
            explanation=f"The {reason} to generate specific filters. Please provide more specific criteria.",
            confidence_score=0.25,
            alternative_suggestions=[
                {
                    "suggestion": "Try specifying exact field names or values",
                    "example": "Instead of 'good products', try 'products with rating > 4'",
                }
            ],
        )

    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """Check if response has low confidence"""
        return self.confidence_score < threshold

    def has_partial_results(self) -> bool:
        """Check if response has partial results (some filters but low confidence)"""
        return len(self.generated_filters) > 0 and self.confidence_score < 0.8

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


class CedarCompletionRequest(BaseModel):
    """Request model for Cedar LangChain completions"""

    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    additionalContext: Optional[Dict[str, Any]] = None
    userId: Optional[str] = None
    threadId: Optional[str] = None


class ChatGenerateTagsRequest(BaseModel):
    """Request model for generating photo tags"""

    imageUrl: str
    selectedTags: List[str] = []
    additionalContext: Optional[Dict[str, Any]] = None
    currentContext: Optional[Dict[str, Any]] = None
    chatHistory: Optional[List[Dict[str, Any]]] = None


class ChatFillTagsRequest(BaseModel):
    """Request model for filling remaining photo tags"""

    currentTags: List[str]
    maxTags: int
    additionalContext: Optional[Dict[str, Any]] = None
    currentContext: Optional[Dict[str, Any]] = None
    chatHistory: Optional[List[Dict[str, Any]]] = None


class ChatGenerateCaptionRequest(BaseModel):
    """Request model for generating photo captions"""

    imageUrl: str
    currentTags: List[str] = []
    filename: Optional[str] = None
    additionalContext: Optional[Dict[str, Any]] = None
    currentContext: Optional[Dict[str, Any]] = None
    chatHistory: Optional[List[Dict[str, Any]]] = None


class ChatFillCaptionRequest(BaseModel):
    """Request model for filling/enhancing photo captions"""

    currentCaption: str
    imageData: Optional[Dict[str, Any]] = None
    imageUrl: Optional[str] = None
    tags: List[str] = []
    additionalContext: Optional[Dict[str, Any]] = None
    currentContext: Optional[Dict[str, Any]] = None
    chatHistory: Optional[List[Dict[str, Any]]] = None
