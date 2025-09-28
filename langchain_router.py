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
from langchain.models import (
    CedarCompletionRequest,
    ChatGenerateTagsRequest,
    ChatFillTagsRequest,
    ChatGenerateCaptionRequest,
    ChatFillCaptionRequest,
)

router = APIRouter(prefix="/langchain", tags=["LangChain"])

# Initialize the LangChain wrapper
langchain_wrapper = LangChainAPIWrapper()


async def create_streaming_response(
    openai_stream, request_id: str
) -> AsyncGenerator[str, None]:
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
                                "content": choice.delta.content
                                if choice.delta and choice.delta.content
                                else ""
                            },
                            "finish_reason": choice.finish_reason,
                        }
                    ],
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
            "error": {"message": str(e), "type": "stream_error"},
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

                        request_id = (
                            f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        )

                        return StreamingResponse(
                            create_streaming_response(openai_stream, request_id),
                            media_type="text/plain",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                                "Content-Type": "text/plain; charset=utf-8",
                            },
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Streaming completion error: {str(e)}",
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="OpenAI client not available for streaming",
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


# Chat endpoints for frontend integration
@router.post("/chat/generate-tags")
async def chat_generate_tags(request: ChatGenerateTagsRequest):
    """Generate photo tags with at least one from required categories"""
    try:
        if not langchain_wrapper.openai_available:
            raise HTTPException(status_code=500, detail="OpenAI client not available")

        # Extract context from cedar_state or legacy fields
        context_info = ""
        if request.cedar_state:
            if request.cedar_state.get('additionalContext'):
                context_info += f"Additional Context: {request.cedar_state['additionalContext']}\n"
            if request.cedar_state.get('currentContext'):
                context_info += f"Current Context: {request.cedar_state['currentContext']}\n"
            if request.cedar_state.get('chatHistory'):
                context_info += f"Chat History: {request.cedar_state['chatHistory']}\n"
        else:
            # Fallback to legacy fields
            if request.additionalContext:
                context_info += f"Additional Context: {request.additionalContext}\n"
            if request.currentContext:
                context_info += f"Current Context: {request.currentContext}\n"
            if request.chatHistory:
                context_info += f"Chat History: {request.chatHistory}\n"

        # Build the prompt with required tag constraint
        current_tags = request.current_tags or request.selectedTags or []
        selected_tags_str = ", ".join(current_tags) if current_tags else "none"

        prompt = f"""
You are an AI assistant helping to generate relevant tags for photos.

IMPORTANT CONSTRAINT: At least one of the tags you generate MUST be from this list: "food", "hacking", or "fun".

Current selected tags: {selected_tags_str}
{context_info}

Looking at this image, generate 5-8 relevant and descriptive tags. The tags should be:
- Descriptive of what's actually in the image
- Relevant for photo organization and searching
- Include at least one tag from: "food", "hacking", or "fun"
- Avoid duplicating already selected tags: {selected_tags_str}

Return the tags as a simple JSON array of strings like: ["tag1", "tag2", "tag3"]
"""

        # Call OpenAI Vision API with base64 or URL
        image_url = request.image_url or request.imageUrl
        result = await langchain_wrapper.call_openai_vision(
            prompt=prompt,
            image_url=image_url,
            image_base64=request.image_base64,
            temperature=0.7,
            max_tokens=200,
        )

        # Parse the response to extract tags array
        import json

        try:
            content = result.get("content", "")
            # Try to extract JSON array from the response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                tags_json = content[start_idx:end_idx]
                tags = json.loads(tags_json)
            else:
                # Fallback: extract comma-separated tags
                tags = [tag.strip().strip('"') for tag in content.split(",")]
                tags = [tag for tag in tags if tag]  # Remove empty strings
        except (json.JSONDecodeError, ValueError, IndexError):
            # Fallback tags if parsing fails
            tags = ["fun", "photo", "memories"]

        # Ensure at least one required tag is included
        required_tags = ["food", "hacking", "fun"]
        has_required = any(
            tag.lower() in [rt.lower() for rt in required_tags] for tag in tags
        )
        if not has_required:
            tags.insert(0, "fun")  # Add "fun" as default required tag

        return JSONResponse(content={"tags": tags})

    except Exception as e:
        logger.error(f"Generate tags error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate tags: {str(e)}"
        )


@router.post("/chat/fill-tags")
async def chat_fill_tags(request: ChatFillTagsRequest):
    """Fill remaining tag slots with relevant suggestions"""
    try:
        if not langchain_wrapper.openai_available:
            raise HTTPException(status_code=500, detail="OpenAI client not available")

        current_tags = request.current_tags or request.currentTags or []
        max_tags = request.max_tags or request.maxTags or 10
        current_tags_str = ", ".join(current_tags)
        remaining_slots = max(0, max_tags - len(current_tags))

        if remaining_slots == 0:
            return JSONResponse(content={"tags": []})

        # Build context information
        context_info = ""
        if request.additionalContext:
            context_info += f"Additional Context: {request.additionalContext}\n"
        if request.currentContext:
            context_info += f"Current Context: {request.currentContext}\n"
        if request.chatHistory:
            context_info += f"Chat History: {request.chatHistory}\n"

        prompt = f"""
You are an AI assistant helping to suggest additional photo tags.

Current tags: {current_tags_str}
Need {remaining_slots} more tags to reach the maximum of {max_tags} tags.
{context_info}

Based on the current tags, suggest {remaining_slots} additional relevant tags that would complement the existing ones. The tags should be:
- Related to the current theme/subject
- Useful for photo organization
- Different from existing tags
- General enough to be useful

IMPORTANT: At least one of your suggestions should be from: "food", "hacking", or "fun" (if not already present in current tags).

Return the tags as a simple JSON array of strings like: ["tag1", "tag2", "tag3"]
"""

        # Call OpenAI Chat API for text completion
        messages = [{"role": "user", "content": prompt}]
        result = await langchain_wrapper.call_openai_chat(
            messages=messages,
            temperature=0.6,
            max_tokens=150,
        )

        # Parse the response
        import json

        try:
            content = result.get("content", "")
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                tags_json = content[start_idx:end_idx]
                tags = json.loads(tags_json)
            else:
                tags = [tag.strip().strip('"') for tag in content.split(",")]
                tags = [tag for tag in tags if tag]
        except (json.JSONDecodeError, ValueError, IndexError):
            # Fallback tags
            required_tags = ["food", "hacking", "fun"]
            current_lower = [t.lower() for t in current_tags]
            fallback_tag = next(
                (tag for tag in required_tags if tag.lower() not in current_lower),
                "memories",
            )
            tags = [fallback_tag]

        # Limit to requested number of tags
        tags = tags[:remaining_slots]

        return JSONResponse(content={"tags": tags})

    except Exception as e:
        logger.error(f"Fill tags error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fill tags: {str(e)}")


@router.post("/chat/generate-caption")
async def chat_generate_caption(request: ChatGenerateCaptionRequest):
    """Generate a caption for the photo"""
    try:
        if not langchain_wrapper.openai_available:
            raise HTTPException(status_code=500, detail="OpenAI client not available")

        # Extract context from cedar_state or legacy fields
        context_info = ""
        if request.cedar_state:
            if request.cedar_state.get('additionalContext'):
                context_info += f"Additional Context: {request.cedar_state['additionalContext']}\n"
            if request.cedar_state.get('currentContext'):
                context_info += f"Current Context: {request.cedar_state['currentContext']}\n"
            if request.cedar_state.get('chatHistory'):
                context_info += f"Chat History: {request.cedar_state['chatHistory']}\n"
        else:
            # Fallback to legacy fields
            if request.additionalContext:
                context_info += f"Additional Context: {request.additionalContext}\n"
            if request.currentContext:
                context_info += f"Current Context: {request.currentContext}\n"
            if request.chatHistory:
                context_info += f"Chat History: {request.chatHistory}\n"

        tags = request.tags or request.currentTags or []
        tags_str = ", ".join(tags) if tags else "none"
        filename_info = f"Filename: {request.filename}" if request.filename else ""

        prompt = f"""
You are an AI assistant helping to generate engaging captions for photos.

Current tags: {tags_str}
{filename_info}
{context_info}

Looking at this image, generate a captivating and descriptive caption that:
- Describes what's happening in the image
- Captures the mood or atmosphere
- Is engaging and social media friendly
- Is 1-2 sentences long
- Incorporates relevant context from the tags when appropriate

Generate just the caption text, nothing else.
"""

        # Call OpenAI Vision API with base64 or URL
        image_url = request.image_url or request.imageUrl
        result = await langchain_wrapper.call_openai_vision(
            prompt=prompt,
            image_url=image_url,
            image_base64=request.image_base64,
            temperature=0.8,
            max_tokens=100,
        )

        caption = result.get("content", "").strip()

        # Clean up the caption (remove quotes if present)
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]

        return JSONResponse(content={"caption": caption})

    except Exception as e:
        logger.error(f"Generate caption error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate caption: {str(e)}"
        )


@router.post("/chat/fill-caption")
async def chat_fill_caption(request: ChatFillCaptionRequest):
    """Enhance or fill out an existing caption"""
    try:
        if not langchain_wrapper.openai_available:
            raise HTTPException(status_code=500, detail="OpenAI client not available")

        # Extract context from cedar_state or legacy fields
        context_info = ""
        if request.cedar_state:
            if request.cedar_state.get('additionalContext'):
                context_info += f"Additional Context: {request.cedar_state['additionalContext']}\n"
            if request.cedar_state.get('currentContext'):
                context_info += f"Current Context: {request.cedar_state['currentContext']}\n"
            if request.cedar_state.get('chatHistory'):
                context_info += f"Chat History: {request.cedar_state['chatHistory']}\n"
        else:
            # Fallback to legacy fields
            if request.additionalContext:
                context_info += f"Additional Context: {request.additionalContext}\n"
            if request.currentContext:
                context_info += f"Current Context: {request.currentContext}\n"
            if request.chatHistory:
                context_info += f"Chat History: {request.chatHistory}\n"

        tags_str = ", ".join(request.tags) if request.tags else "none"

        prompt = f"""
You are an AI assistant helping to enhance photo captions.

Current caption: "{request.current_caption or request.currentCaption}"
Tags: {tags_str}
{context_info}

Please enhance this caption by:
- Making it more engaging and descriptive
- Adding relevant details that complement the existing caption
- Keeping the original tone and style
- Making it 1-3 sentences long
- Incorporating relevant context from tags when appropriate

If the current caption is already good, you can make minor improvements or keep it mostly the same.

Return just the enhanced caption text, nothing else.
"""

        # Use image URL or base64 if provided for visual context
        image_url = request.image_url or request.imageUrl
        if image_url or request.image_base64:
            result = await langchain_wrapper.call_openai_vision(
                prompt=prompt,
                image_url=image_url,
                image_base64=request.image_base64,
                temperature=0.7,
                max_tokens=150,
            )
        else:
            # Text-only enhancement
            messages = [{"role": "user", "content": prompt}]
            result = await langchain_wrapper.call_openai_chat(
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )

        caption = result.get("content", "").strip()

        # Clean up the caption
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]

        return JSONResponse(content={"caption": caption})

    except Exception as e:
        logger.error(f"Fill caption error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to enhance caption: {str(e)}"
        )
