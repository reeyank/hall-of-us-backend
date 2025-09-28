# main.py
from loguru import logger
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
import base64
import json
from datetime import datetime

from langchain import (
    image_tagging_api,
    filter_generation_api,
    ImageTaggingRequest,
    FilterGenerationRequest,
    APIResponse,
)
from langchain.wrapper import LangChainAPIWrapper
from langchain.models import CedarCompletionRequest

router = APIRouter(prefix="/langchain", tags=["LangChain"])

# Initialize the LangChain wrapper
langchain_wrapper = LangChainAPIWrapper()


async def create_streaming_response(openai_stream, request_id: str) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Server-Sent Events format"""
    try:
        async for chunk in openai_stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]

                # Create the streaming response chunk in OpenAI format
                response_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {
                                "content": choice.delta.content if choice.delta and choice.delta.content else ""
                            },
                            "finish_reason": choice.finish_reason
                        }
                    ]
                }

                # Send the chunk in SSE format
                yield f"data: {json.dumps(response_chunk)}\n\n"

                # If this is the last chunk, send the done signal
                if choice.finish_reason:
                    yield "data: [DONE]\n\n"
                    break
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        error_chunk = {
            "id": request_id,
            "object": "error",
            "error": {"message": str(e), "type": "stream_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


def extract_image_from_context(additional_context: Any) -> str | None:
    """Extract image URL from additionalContext if available"""
    if not additional_context:
        return None

    # Look for uploaded images in the context
    for key, value in additional_context.items():
        if key.startswith("uploaded_image_") and isinstance(value, dict):
            image_data = value.get("data", {})
            if isinstance(image_data, dict) and "url" in image_data:
                return image_data["url"]

    return None


@router.post("/chat/completions")
@router.post("/completions")  # Legacy endpoint for backward compatibility
async def completions(request: CedarCompletionRequest):
    """
    Main completions endpoint for LangChain provider

    Supports both regular completions and structured output.
    Also supports streaming when stream=true is passed.
    """
    try:
        # Validate the request
        if not request.validate_request():
            logger.info("Invalid request data")
            raise HTTPException(status_code=400, detail="Invalid request")

        logger.info(f"Received completion request: {request}")
        # Check if there's an uploaded image to analyze
        image_url = extract_image_from_context(request.additionalContext)

        if image_url:
            # Extract the user's question from messages
            user_messages = [
                msg for msg in request.messages if msg.get("role") == "user"
            ]
            user_query = user_messages[-1].get("content", "") if user_messages else ""

            # Use image analysis
            if langchain_wrapper.openai_available:
                try:
                    result = await langchain_wrapper.call_openai_vision(
                        prompt=user_query,
                        image_url=image_url,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    )

                    return JSONResponse(
                        content={
                            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "object": "chat.completion",
                            "created": int(datetime.now().timestamp()),
                            "model": "gpt-4o",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result.get("content", ""),
                                    },
                                    "finish_reason": result.get(
                                        "finish_reason", "stop"
                                    ),
                                }
                            ],
                            "usage": result.get(
                                "usage",
                                {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                },
                            ),
                            "image_analysis": True,
                        },
                        status_code=200,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Image analysis error: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )
        else:
            # Regular text completion
            messages = request.messages
            temperature = request.temperature
            max_tokens = request.max_tokens
            stream = request.stream
            logger.info(f"Processing text completion, stream={stream}")

            if stream:
                # Handle streaming response
                if langchain_wrapper.openai_available:
                    try:
                        openai_stream = await langchain_wrapper.call_openai_chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                        )

                        request_id = f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                        return StreamingResponse(
                            create_streaming_response(openai_stream, request_id),
                            media_type="text/plain",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                                "Content-Type": "text/plain; charset=utf-8",
                            }
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, detail=f"Streaming completion error: {str(e)}"
                        )
                else:
                    raise HTTPException(
                        status_code=500, detail="OpenAI client not available for streaming"
                    )
            else:
                # Use regular completion
                if langchain_wrapper.openai_available:
                    try:
                        result = await langchain_wrapper.call_openai_chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        return JSONResponse(
                            content={
                                "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                "object": "chat.completion",
                                "created": int(datetime.now().timestamp()),
                                "model": result.get("model", "gpt-4"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": result.get("content", ""),
                                        },
                                        "finish_reason": result.get(
                                            "finish_reason", "stop"
                                        ),
                                    }
                                ],
                                "usage": result.get(
                                    "usage",
                                    {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0,
                                    },
                                ),
                            },
                            status_code=200,
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, detail=f"Completion error: {str(e)}"
                        )
                else:
                    raise HTTPException(
                        status_code=500, detail="OpenAI client not available"
                    )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/")
async def read_root():
    return {"message": "Hall of Us API - LangChain powered backend"}


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "hall-of-us-backend"}


# Image Tagging Endpoints
@router.post("/api/v1/image/tags/from-url")
async def generate_tags_from_url(request: ImageTaggingRequest):
    """Generate tags from an image URL"""
    try:
        response: APIResponse = await image_tagging_api.generate_tags_from_url(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict()
                    if hasattr(response.data, "dict")
                    else response.data,
                    "execution_time_ms": response.execution_time_ms,
                },
                status_code=200,
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/v1/image/tags/from-base64")
async def generate_tags_from_base64(request: ImageTaggingRequest):
    """Generate tags from a base64 encoded image"""
    try:
        response: APIResponse = await image_tagging_api.generate_tags_from_base64(
            request
        )

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict()
                    if hasattr(response.data, "dict")
                    else response.data,
                    "execution_time_ms": response.execution_time_ms,
                },
                status_code=200,
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/v1/image/tags/from-file")
async def generate_tags_from_file(
    file: UploadFile = File(...), max_tags: int = 10, confidence_threshold: float = 0.5
):
    """Generate tags from an uploaded image file"""
    try:
        # Read file and convert to base64
        file_content = await file.read()
        base64_image = base64.b64encode(file_content).decode("utf-8")

        request = ImageTaggingRequest(
            image_base64=base64_image,
            max_tags=max_tags,
            confidence_threshold=confidence_threshold,
        )

        response: APIResponse = await image_tagging_api.generate_tags_from_base64(
            request
        )

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict()
                    if hasattr(response.data, "dict")
                    else response.data,
                    "execution_time_ms": response.execution_time_ms,
                    "file_info": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size_bytes": len(file_content),
                    },
                },
                status_code=200,
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Filter Generation Endpoints
@router.post("/api/v1/filters/generate")
async def generate_filters(request: FilterGenerationRequest):
    """Generate filter configuration from natural language query"""
    try:
        response: APIResponse = await filter_generation_api.generate_filters(request)

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data.dict()
                    if hasattr(response.data, "dict")
                    else response.data,
                    "execution_time_ms": response.execution_time_ms,
                },
                status_code=200,
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/v1/filters/validate")
async def validate_filters(
    filters: List[Dict[str, Any]], available_filters: List[Dict[str, Any]]
):
    """Validate generated filters against available filter definitions"""
    try:
        response: APIResponse = await filter_generation_api.validate_filters(
            filters, available_filters
        )

        if response.success:
            return JSONResponse(
                content={
                    "success": True,
                    "data": response.data,
                    "execution_time_ms": response.execution_time_ms,
                },
                status_code=200,
            )
        else:
            raise HTTPException(status_code=400, detail=response.error)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Example endpoint for testing
@router.post("/api/v1/test/image-tagging")
async def test_image_tagging():
    """Test endpoint for image tagging with sample data"""
    sample_request = ImageTaggingRequest(
        image_url="https://example.com/sample-image.jpg",
        max_tags=5,
        confidence_threshold=0.7,
    )

    return await generate_tags_from_url(sample_request)


@router.post("/api/v1/test/filter-generation")
async def test_filter_generation():
    """Test endpoint for filter generation with sample data"""
    sample_request = FilterGenerationRequest(
        natural_language_query="Show me recent popular items in technology category",
        available_filters=[
            {"field": "created_at", "type": "date", "operators": [">=", "<=", "=="]},
            {
                "field": "popularity_score",
                "type": "numeric",
                "operators": [">=", "<=", "=="],
            },
            {
                "field": "category",
                "type": "categorical",
                "operators": ["in", "==", "!="],
            },
            {
                "field": "title",
                "type": "text",
                "operators": ["contains", "==", "startswith"],
            },
        ],
        max_filters=3,
    )

    return await generate_filters(sample_request)
