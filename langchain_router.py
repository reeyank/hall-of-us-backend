# main.py
from loguru import logger
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime

from langchain import (
    LangChainAPIWrapper,
    create_streaming_response,
    create_chat_completion_response,
    create_vision_response,
    extract_image_from_context,
    ChatHandlers,
)
from langchain.models import (
    CedarCompletionRequest,
    ChatGenerateTagsRequest,
    ChatFillTagsRequest,
    ChatGenerateCaptionRequest,
    ChatFillCaptionRequest,
)

router = APIRouter(prefix="/langchain", tags=["LangChain"])

# Initialize the LangChain wrapper and chat handlers
langchain_wrapper = LangChainAPIWrapper()
chat_handlers = ChatHandlers(langchain_wrapper)


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
                        content=create_vision_response(result),
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
                            content=create_chat_completion_response(result),
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


# Chat endpoints for frontend integration
@router.post("/chat/generate-tags")
async def chat_generate_tags(request: ChatGenerateTagsRequest):
    """Generate photo tags with at least one from required categories"""
    return await chat_handlers.generate_tags(request)


@router.post("/chat/fill-tags")
async def chat_fill_tags(request: ChatFillTagsRequest):
    """Fill remaining tag slots with relevant suggestions"""
    return await chat_handlers.fill_tags(request)


@router.post("/chat/generate-caption")
async def chat_generate_caption(request: ChatGenerateCaptionRequest):
    """Generate a caption for the photo"""
    return await chat_handlers.generate_caption(request)


@router.post("/chat/fill-caption")
async def chat_fill_caption(request: ChatFillCaptionRequest):
    """Enhance or fill out an existing caption"""
    return await chat_handlers.fill_caption(request)
