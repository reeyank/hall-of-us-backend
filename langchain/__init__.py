"""
LangChain module initialization
"""

from .wrapper import api_wrapper
from .image_tagging import ImageTaggingAPISet, ImageTaggingRequest, ImageTaggingResponse
from .filter_generation import FilterGenerationAPISet, FilterGenerationRequest, FilterGenerationResponse
from .models import APIResponse

# Initialize global instances
image_tagging_api = ImageTaggingAPISet(api_wrapper)
filter_generation_api = FilterGenerationAPISet(api_wrapper)

__all__ = [
    'api_wrapper',
    'image_tagging_api',
    'filter_generation_api',
    'ImageTaggingAPISet',
    'ImageTaggingRequest',
    'ImageTaggingResponse',
    'FilterGenerationAPISet',
    'FilterGenerationRequest',
    'FilterGenerationResponse',
    'APIResponse'
]
