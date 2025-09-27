# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import base64

from langchain import (
    image_tagging_api,
    filter_generation_api,
    ImageTaggingRequest,
    FilterGenerationRequest,
    APIResponse
)

app = FastAPI(
    title="Hall of Us API",
    description="LangChain-powered API for image tagging and filter generation",
    version="1.0.0"
)

@app.get("/")
async def read_root():
    return {"message": "Hall of Us API - LangChain powered backend"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "hall-of-us-backend"}

# Image Tagging Endpoints
@app.post("/api/v1/image/tags/from-url")
async def generate_tags_from_url(request: ImageTaggingRequest):
    """Generate tags from an image URL"""
    try:
        response: APIResponse = await image_tagging_api.generate_tags_from_url(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict() if hasattr(response.data, 'dict') else response.data,
                    "execution_time_ms": response.execution_time_ms
                },
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/image/tags/from-base64")
async def generate_tags_from_base64(request: ImageTaggingRequest):
    """Generate tags from a base64 encoded image"""
    try:
        response: APIResponse = await image_tagging_api.generate_tags_from_base64(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict() if hasattr(response.data, 'dict') else response.data,
                    "execution_time_ms": response.execution_time_ms
                },
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/image/tags/from-file")
async def generate_tags_from_file(file: UploadFile = File(...), max_tags: int = 10, confidence_threshold: float = 0.5):
    """Generate tags from an uploaded image file"""
    try:
        # Read file and convert to base64
        file_content = await file.read()
        base64_image = base64.b64encode(file_content).decode('utf-8')

        request = ImageTaggingRequest(
            image_base64=base64_image,
            max_tags=max_tags,
            confidence_threshold=confidence_threshold
        )

        response: APIResponse = await image_tagging_api.generate_tags_from_base64(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict() if hasattr(response.data, 'dict') else response.data,
                    "execution_time_ms": response.execution_time_ms,
                    "file_info": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size_bytes": len(file_content)
                    }
                },
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Filter Generation Endpoints
@app.post("/api/v1/filters/generate")
async def generate_filters(request: FilterGenerationRequest):
    """Generate filter configuration from natural language query"""
    try:
        response: APIResponse = await filter_generation_api.generate_filters(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict() if hasattr(response.data, 'dict') else response.data,
                    "execution_time_ms": response.execution_time_ms
                },
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/filters/validate")
async def validate_filters(
    filters: List[Dict[str, Any]],
    available_filters: List[Dict[str, Any]]
):
    """Validate generated filters against available filter definitions"""
    try:
        response: APIResponse = await filter_generation_api.validate_filters(filters, available_filters)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data,
                    "execution_time_ms": response.execution_time_ms
                },
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Example endpoint for testing
@app.post("/api/v1/test/image-tagging")
async def test_image_tagging():
    """Test endpoint for image tagging with sample data"""
    sample_request = ImageTaggingRequest(
        image_url="https://example.com/sample-image.jpg",
        max_tags=5,
        confidence_threshold=0.7
    )

    return await generate_tags_from_url(sample_request)

@app.post("/api/v1/test/filter-generation")
async def test_filter_generation():
    """Test endpoint for filter generation with sample data"""
    sample_request = FilterGenerationRequest(
        natural_language_query="Show me recent popular items in technology category",
        available_filters=[
            {"field": "created_at", "type": "date", "operators": [">=", "<=", "=="]},
            {"field": "popularity_score", "type": "numeric", "operators": [">=", "<=", "=="]},
            {"field": "category", "type": "categorical", "operators": ["in", "==", "!="]},
            {"field": "title", "type": "text", "operators": ["contains", "==", "startswith"]}
        ],
        max_filters=3
    )

    return await generate_filters(sample_request)
