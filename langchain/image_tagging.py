"""
Image Tagging API Set

Specialized API set for image tagging operations.
This handles various image input formats and provides structured tagging responses.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .models import APIResponse, ImageTaggingRequest, ImageTaggingResponse
from .wrapper import LangChainAPIWrapper
from .logger import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class ImageTaggingAPISet:
    """
    Specialized API set for image tagging operations.

    This handles various image input formats and provides structured tagging responses.
    """

    def __init__(self, wrapper: LangChainAPIWrapper):
        self.wrapper = wrapper

    async def generate_tags_from_url(self, request: ImageTaggingRequest) -> APIResponse:
        """Generate tags from image URL"""
        if not request.image_url:
            return APIResponse(
                success=False,
                error="image_url is required for this method",
                timestamp=datetime.now(),
            )

        # Create the chain for image tagging
        chain = self.wrapper.create_simple_chain(self._process_image_url_tagging)

        return await self.wrapper.execute_chain(chain, request.dict())

    async def generate_tags_from_base64(
        self, request: ImageTaggingRequest
    ) -> APIResponse:
        """Generate tags from base64 encoded image"""
        if not request.image_base64:
            return APIResponse(
                success=False,
                error="image_base64 is required for this method",
                timestamp=datetime.now(),
            )

        chain = self.wrapper.create_simple_chain(self._process_image_base64_tagging)

        return await self.wrapper.execute_chain(chain, request.dict())

    async def generate_tags_from_file(
        self, request: ImageTaggingRequest
    ) -> APIResponse:
        """Generate tags from local image file"""
        if not request.image_path:
            return APIResponse(
                success=False,
                error="image_path is required for this method",
                timestamp=datetime.now(),
            )

        chain = self.wrapper.create_simple_chain(self._process_image_file_tagging)

        return await self.wrapper.execute_chain(chain, request.dict())

    async def _process_image_url_tagging(
        self, inputs: Dict[str, Any]
    ) -> ImageTaggingResponse:
        """Process image from URL using OpenAI Vision API"""

        # Use OpenAI Vision API
        assert self.wrapper.openai_available is not None

        prompt = self._build_tagging_prompt(inputs)

        openai_response = await self.wrapper.call_openai_vision(
            prompt=prompt,
            image_url=inputs.get("image_url"),
            model="gpt-4o",
            max_tokens=500,
        )

        # Parse OpenAI response into structured tags
        tags = self._parse_openai_tags_response(
            openai_response["content"],
            inputs.get("max_tags", 10),
            inputs.get("confidence_threshold", 0.5),
        )

        return ImageTaggingResponse(
            tags=tags,
            image_metadata={
                "source": "url",
                "url": inputs.get("image_url"),
                "processed_at": datetime.now().isoformat(),
                "model_used": openai_response["model"],
            },
            processing_info={
                "model": openai_response["model"],
                "confidence_threshold": inputs.get("confidence_threshold", 0.5),
                "tokens_used": openai_response["usage"]["total_tokens"],
                "method": "openai_vision",
            },
        )

    async def _process_image_base64_tagging(
        self, inputs: Dict[str, Any]
    ) -> ImageTaggingResponse:
        """Process image from base64 using OpenAI Vision API"""

        # Use OpenAI Vision API
        assert self.wrapper.openai_available is not None

        prompt = self._build_tagging_prompt(inputs)

        openai_response = await self.wrapper.call_openai_vision(
            prompt=prompt,
            image_base64=inputs.get("image_base64"),
            model="gpt-4o",
            max_tokens=500,
        )

        # Parse OpenAI response into structured tags
        tags = self._parse_openai_tags_response(
            openai_response["content"],
            inputs.get("max_tags", 10),
            inputs.get("confidence_threshold", 0.5),
        )

        return ImageTaggingResponse(
            tags=tags,
            image_metadata={
                "source": "base64",
                "size_bytes": len(inputs.get("image_base64", "")),
                "processed_at": datetime.now().isoformat(),
                "model_used": openai_response["model"],
            },
            processing_info={
                "model": openai_response["model"],
                "confidence_threshold": inputs.get("confidence_threshold", 0.5),
                "tokens_used": openai_response["usage"]["total_tokens"],
                "method": "openai_vision",
            },
        )

    async def _process_image_file_tagging(
        self, inputs: Dict[str, Any]
    ) -> ImageTaggingResponse:
        """Process image from file path - stub implementation"""
        # TODO: Integrate with your actual image tagging service

        await asyncio.sleep(0.3)

        mock_tags = [
            {"tag": "landscape", "confidence": 0.89, "category": "composition"},
            {"tag": "nature", "confidence": 0.94, "category": "environment"},
            {"tag": "outdoor", "confidence": 0.87, "category": "setting"},
        ]

        return ImageTaggingResponse(
            tags=mock_tags[: inputs.get("max_tags", 10)],
            image_metadata={
                "source": "file",
                "file_path": inputs.get("image_path"),
                "processed_at": datetime.now().isoformat(),
            },
        )

    def _build_tagging_prompt(self, inputs: Dict[str, Any]) -> str:
        """Build a prompt for OpenAI Vision API to generate image tags"""
        max_tags = inputs.get("max_tags", 10)
        confidence_threshold = inputs.get("confidence_threshold", 0.5)
        categories = inputs.get("categories", [])

        prompt = f"""Analyze this image and generate up to {max_tags} descriptive tags.

For each tag, provide:
- The tag name
- A confidence score (0.0-1.0, only include tags with confidence >= {confidence_threshold})
- A category for the tag (e.g., 'object', 'person', 'scene', 'color', 'style', 'emotion', 'activity')

Format your response as a JSON array of objects like this:
[
  {{"tag": "example_tag", "confidence": 0.95, "category": "object"}},
  {{"tag": "another_tag", "confidence": 0.82, "category": "scene"}}
]

"""

        if categories:
            prompt += (
                f"\nFocus particularly on these categories: {', '.join(categories)}\n"
            )

        prompt += (
            "\nBe specific and descriptive. Only include tags you are confident about."
        )

        return prompt

    def _parse_openai_tags_response(
        self, response_content: str, max_tags: int, confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Parse OpenAI response content into structured tags"""
        import json
        import re

        # Extract JSON from the response
        json_match = re.search(r"\[.*\]", response_content, re.DOTALL)
        json_str = json_match.group()
        tags_data = json.loads(json_str)

        # Filter and validate tags
        valid_tags = []
        for tag_data in tags_data:
            if (
                isinstance(tag_data, dict)
                and "tag" in tag_data
                and "confidence" in tag_data
                and tag_data.get("confidence", 0) >= confidence_threshold
            ):
                valid_tags.append(
                    {
                        "tag": str(tag_data["tag"]).lower().strip(),
                        "confidence": float(tag_data["confidence"]),
                        "category": tag_data.get("category", "general"),
                    }
                )

        # Sort by confidence and limit results
        valid_tags.sort(key=lambda x: x["confidence"], reverse=True)
        return valid_tags[:max_tags]
