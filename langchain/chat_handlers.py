from loguru import logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import json

from .wrapper import LangChainAPIWrapper
from .models import (
    ChatGenerateTagsRequest,
    ChatFillTagsRequest,
    ChatGenerateCaptionRequest,
    ChatFillCaptionRequest,
)


class ChatHandlers:
    def __init__(self, langchain_wrapper: LangChainAPIWrapper):
        self.langchain_wrapper = langchain_wrapper

    async def generate_tags(self, request: ChatGenerateTagsRequest):
        """Generate photo tags with at least one from required categories"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(status_code=500, detail="OpenAI client not available")

            context_info = self._build_context_info(request)
            selected_tags_str = (
                ", ".join(request.selectedTags) if request.selectedTags else "none"
            )

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

            result = await self.langchain_wrapper.call_openai_vision(
                prompt=prompt,
                image_url=request.image_url if request.image_url else None,
                temperature=0.7,
                max_tokens=200,
            )

            tags = self._parse_tags_response(result.get("content", ""))
            tags = self._ensure_required_tags(tags)

            return JSONResponse(content={"tags": tags})

        except Exception as e:
            logger.error(f"Generate tags error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate tags: {str(e)}"
            )

    async def fill_tags(self, request: ChatFillTagsRequest):
        """Fill remaining tag slots with relevant suggestions"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(status_code=500, detail="OpenAI client not available")

            current_tags = request.current_tags or request.currentTags
            current_tags_str = ", ".join(current_tags)
            max_tags = request.max_tags or request.maxTags
            remaining_slots = max(0, max_tags - len(current_tags))

            if remaining_slots == 0:
                return JSONResponse(content={"tags": []})

            context_info = self._build_context_info(request)

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

            messages = [{"role": "user", "content": prompt}]
            result = await self.langchain_wrapper.call_openai_chat(
                messages=messages,
                temperature=0.6,
                max_tokens=150,
            )

            tags = self._parse_tags_response(result.get("content", ""))
            tags = tags[:remaining_slots]

            return JSONResponse(content={"tags": tags})

        except Exception as e:
            logger.error(f"Fill tags error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fill tags: {str(e)}")

    async def generate_caption(self, request: ChatGenerateCaptionRequest):
        """Generate a caption for the photo"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(status_code=500, detail="OpenAI client not available")

            context_info = self._build_context_info(request)
            tags_str = ", ".join(request.currentTags) if request.currentTags else "none"
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

            if not request.image_url:
                raise HTTPException(
                    status_code=400, detail="image_url is required for vision API calls"
                )

            result = await self.langchain_wrapper.call_openai_vision(
                prompt=prompt,
                image_url=request.image_url,
                temperature=0.8,
                max_tokens=100,
            )

            caption = self._clean_caption(result.get("content", "").strip())
            return JSONResponse(content={"caption": caption})

        except Exception as e:
            logger.error(f"Generate caption error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate caption: {str(e)}"
            )

    async def fill_caption(self, request: ChatFillCaptionRequest):
        """Enhance or fill out an existing caption"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(status_code=500, detail="OpenAI client not available")

            context_info = self._build_context_info(request)
            tags_str = ", ".join(request.tags) if request.tags else "none"

            prompt = f"""
You are an AI assistant helping to enhance photo captions.

Current caption: "{request.currentCaption}"
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

            if request.image_url:
                result = await self.langchain_wrapper.call_openai_vision(
                    prompt=prompt,
                    image_url=request.image_url,
                    temperature=0.7,
                    max_tokens=150,
                )
            else:
                messages = [{"role": "user", "content": prompt}]
                result = await self.langchain_wrapper.call_openai_chat(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150,
                )

            caption = self._clean_caption(result.get("content", "").strip())
            return JSONResponse(content={"caption": caption})

        except Exception as e:
            logger.error(f"Fill caption error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to enhance caption: {str(e)}"
            )

    def _build_context_info(self, request) -> str:
        """Build context information from request"""
        context_info = ""
        if hasattr(request, 'additionalContext') and request.additionalContext:
            context_info += f"Additional Context: {request.additionalContext}\n"
        if hasattr(request, 'currentContext') and request.currentContext:
            context_info += f"Current Context: {request.currentContext}\n"
        if hasattr(request, 'chatHistory') and request.chatHistory:
            context_info += f"Chat History: {request.chatHistory}\n"
        return context_info

    def _parse_tags_response(self, content: str) -> List[str]:
        """Parse tags from AI response"""
        try:
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                tags_json = content[start_idx:end_idx]
                tags = json.loads(tags_json)
            else:
                tags = [tag.strip().strip('"') for tag in content.split(",")]
                tags = [tag for tag in tags if tag]
        except (json.JSONDecodeError, ValueError, IndexError):
            tags = ["fun", "photo", "memories"]
        return tags

    def _ensure_required_tags(self, tags: List[str]) -> List[str]:
        """Ensure at least one required tag is included"""
        required_tags = ["food", "hacking", "fun"]
        has_required = any(
            tag.lower() in [rt.lower() for rt in required_tags] for tag in tags
        )
        if not has_required:
            tags.insert(0, "fun")
        return tags

    def _clean_caption(self, caption: str) -> str:
        """Clean up caption by removing quotes"""
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        return caption