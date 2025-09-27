# Hall of Us Backend - LangChain API Wrapper

A FastAPI backend with LangChain integration for specialized API operations including image tagging and filter generation.

## Features

### 🏷️ Image Tagging API Set

- **Generate tags from image URLs** - Analyze images hosted online
- **Generate tags from base64 images** - Process uploaded image data
- **Generate tags from file uploads** - Handle direct file uploads
- Configurable confidence thresholds and tag limits
- Structured response format with metadata

### 🔍 Filter Generation API Set

- **Natural language to filters** - Convert queries like "recent popular tech items" into structured filters
- **Filter validation** - Validate generated filters against your available filter definitions
- **Smart suggestions** - Get alternative filter suggestions
- Support for various filter types: date ranges, categories, text search, numeric ranges

### 🔧 LangChain Wrapper

- **Unified API interface** - All API calls go through a consistent wrapper
- **Error handling & retries** - Automatic retry logic with exponential backoff
- **Monitoring & logging** - Built-in execution time tracking and error logging
- **Batch processing** - Execute multiple API calls in parallel
- **Timeout management** - Configurable timeouts for all operations

## API Endpoints

### Image Tagging

```
POST /api/v1/image/tags/from-url      # Tag image from URL
POST /api/v1/image/tags/from-base64   # Tag image from base64
POST /api/v1/image/tags/from-file     # Tag uploaded image file
```

### Filter Generation

```
POST /api/v1/filters/generate         # Generate filters from natural language
POST /api/v1/filters/validate         # Validate filter configurations
```

### Testing

```
POST /api/v1/test/image-tagging       # Test image tagging with sample data
POST /api/v1/test/filter-generation   # Test filter generation with sample data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

### 3. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

Or run the examples:

```bash
python examples.py
```

## Example Usage

### Image Tagging

```python
from langchain_wrapper import image_tagging_api, ImageTaggingRequest

# Tag an image from URL
request = ImageTaggingRequest(
    image_url="https://example.com/image.jpg",
    max_tags=5,
    confidence_threshold=0.7
)

response = await image_tagging_api.generate_tags_from_url(request)
print(response.data.tags)
```

### Filter Generation

```python
from langchain_wrapper import filter_generation_api, FilterGenerationRequest

# Generate filters from natural language
request = FilterGenerationRequest(
    natural_language_query="Show me recent popular tech articles",
    available_filters=[
        {"field": "created_at", "type": "date", "operators": [">=", "<="]},
        {"field": "category", "type": "categorical", "operators": ["==", "in"]},
        {"field": "popularity_score", "type": "numeric", "operators": [">=", "<="]}
    ],
    max_filters=3
)

response = await filter_generation_api.generate_filters(request)
print(response.data.generated_filters)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI App (main.py)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │ Image Tagging   │    │ Filter Generation        │   │
│  │ API Set         │    │ API Set                  │   │
│  └─────────────────┘    └──────────────────────────┘   │
│           │                           │                │
│           └───────────┬───────────────┘                │
│                       │                                │
│              ┌─────────────────┐                       │
│              │ LangChain       │                       │
│              │ API Wrapper     │                       │
│              └─────────────────┘                       │
│                       │                                │
├─────────────────────────────────────────────────────────┤
│         External APIs (OpenAI, Google Vision, etc.)    │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Integrate Real APIs**: Replace the stub implementations with actual API calls:

   - OpenAI Vision API for image tagging
   - OpenAI GPT for natural language processing
   - Google Vision API as alternative

2. **Add Authentication**: Implement API key management and user authentication

3. **Add Caching**: Implement Redis caching for expensive operations

4. **Add Rate Limiting**: Prevent API abuse with rate limiting

5. **Add Database Integration**: Store results and user preferences

6. **Add More API Sets**: Create specialized wrappers for other operations

## Configuration

The system is designed to be easily configurable. Key configuration points:

- **Timeout settings** - Adjust API call timeouts
- **Retry logic** - Configure max retries and backoff strategy
- **Logging levels** - Control verbosity of logging
- **Model selection** - Choose which AI models to use for different tasks

## Error Handling

All API calls return a standardized `APIResponse` format:

```python
{
    "success": bool,
    "data": Any,           # Present when success=True
    "error": str,          # Present when success=False
    "timestamp": datetime,
    "execution_time_ms": float
}
```

This ensures consistent error handling across all API operations.
