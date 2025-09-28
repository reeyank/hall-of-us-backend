from datetime import datetime
from typing import Dict, Any


def create_chat_completion_response(result: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
    """Create a standardized chat completion response"""
    return {
        "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": result.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("content", ""),
                },
                "finish_reason": result.get("finish_reason", "stop"),
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
    }


def create_vision_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a response for vision API calls"""
    response = create_chat_completion_response(result, "gpt-4o")
    response["image_analysis"] = True
    return response


def extract_image_from_context(additional_context: Any) -> str | None:
    """Extract image URL from additionalContext if available"""
    if not additional_context:
        return None

    for key, value in additional_context.items():
        if key.startswith("uploaded_image_") and isinstance(value, dict):
            image_data = value.get("data", {})
            if isinstance(image_data, dict) and "url" in image_data:
                return image_data["url"]

    return None