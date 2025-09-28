from loguru import logger
from typing import AsyncGenerator
import json
from datetime import datetime


async def create_streaming_response(
    openai_stream, request_id: str
) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Server-Sent Events format"""
    try:
        async for chunk in openai_stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]

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

                yield f"data: {json.dumps(response_chunk)}\n\n"

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