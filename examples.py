"""
Usage Examples for Hall of Us API

This file demonstrates how to use the different API sets for image tagging and filter generation.
"""

import asyncio
from langchain import (
    image_tagging_api,
    filter_generation_api,
    ImageTaggingRequest,
    FilterGenerationRequest
)


async def example_image_tagging():
    """Example of using the image tagging API"""
    print("=== Image Tagging Examples ===\n")
    
    # Example 1: Image from URL
    print("1. Tagging image from URL:")
    url_request = ImageTaggingRequest(
        image_url="https://example.com/sample-image.jpg",
        max_tags=5,
        confidence_threshold=0.7
    )
    
    response = await image_tagging_api.generate_tags_from_url(url_request)
    print(f"Success: {response.success}")
    if response.success:
        print(f"Tags: {response.data.tags}")
        print(f"Execution time: {response.execution_time_ms}ms")
    else:
        print(f"Error: {response.error}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Image from base64
    print("2. Tagging image from base64:")
    # This is a small 1x1 transparent PNG in base64
    sample_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    base64_request = ImageTaggingRequest(
        image_base64=sample_base64,
        max_tags=3,
        confidence_threshold=0.5
    )
    
    response = await image_tagging_api.generate_tags_from_base64(base64_request)
    print(f"Success: {response.success}")
    if response.success:
        print(f"Tags: {response.data.tags}")
    else:
        print(f"Error: {response.error}")


async def example_filter_generation():
    """Example of using the filter generation API"""
    print("\n=== Filter Generation Examples ===\n")
    
    # Example available filters (what your system supports)
    available_filters = [
        {
            "field": "created_at",
            "type": "date",
            "operators": [">=", "<=", "=="],
            "display_name": "Creation Date"
        },
        {
            "field": "popularity_score",
            "type": "numeric",
            "operators": [">=", "<=", "=="],
            "display_name": "Popularity Score"
        },
        {
            "field": "category",
            "type": "categorical",
            "operators": ["in", "==", "!="],
            "display_name": "Category",
            "values": ["technology", "science", "art", "music", "sports"]
        },
        {
            "field": "title",
            "type": "text",
            "operators": ["contains", "==", "startswith"],
            "display_name": "Title"
        },
        {
            "field": "tags",
            "type": "array",
            "operators": ["contains_any", "contains_all"],
            "display_name": "Tags"
        }
    ]
    
    # Example 1: Recent popular tech items
    print("1. Query: 'Show me recent popular items in technology category'")
    request1 = FilterGenerationRequest(
        natural_language_query="Show me recent popular items in technology category",
        available_filters=available_filters,
        max_filters=3
    )
    
    response = await filter_generation_api.generate_filters(request1)
    print(f"Success: {response.success}")
    if response.success:
        print(f"Generated filters: {response.data.generated_filters}")
        print(f"Explanation: {response.data.explanation}")
        print(f"Confidence: {response.data.confidence_score}")
    else:
        print(f"Error: {response.error}")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: More complex query
    print("2. Query: 'Find trending music content from the last month'")
    request2 = FilterGenerationRequest(
        natural_language_query="Find trending music content from the last month",
        available_filters=available_filters,
        max_filters=4
    )
    
    response = await filter_generation_api.generate_filters(request2)
    print(f"Success: {response.success}")
    if response.success:
        print(f"Generated filters: {response.data.generated_filters}")
        print(f"Explanation: {response.data.explanation}")
    else:
        print(f"Error: {response.error}")


async def example_filter_validation():
    """Example of validating generated filters"""
    print("\n=== Filter Validation Example ===\n")
    
    available_filters = [
        {"field": "created_at", "type": "date"},
        {"field": "category", "type": "categorical"},
        {"field": "popularity_score", "type": "numeric"}
    ]
    
    # Test filters (mix of valid and invalid)
    test_filters = [
        {
            "type": "date_range",
            "field": "created_at",  # Valid field
            "operator": ">=",
            "value": "2024-01-01"
        },
        {
            "type": "categorical",
            "field": "category",  # Valid field
            "operator": "==",
            "value": "technology"
        },
        {
            "type": "text_search",
            "field": "description",  # Invalid field (not in available_filters)
            "operator": "contains",
            "value": "test"
        }
    ]
    
    response = await filter_generation_api.validate_filters(test_filters, available_filters)
    print(f"Success: {response.success}")
    if response.success:
        print(f"Valid filters: {response.data['valid_filters']}")
        print(f"Invalid filters: {response.data['invalid_filters']}")
        print(f"Summary: {response.data['validation_summary']}")
    else:
        print(f"Error: {response.error}")


async def main():
    """Run all examples"""
    await example_image_tagging()
    await example_filter_generation()
    await example_filter_validation()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())