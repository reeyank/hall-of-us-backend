from loguru import logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import json

from .wrapper import LangChainAPIWrapper
from .shared_context import ChatContext
from .models import (
    ChatGenerateTagsRequest,
    ChatFillTagsRequest,
    ChatGenerateCaptionRequest,
    ChatFillCaptionRequest,
    ChatFilterImagesRequest,
)


class ChatHandlers:
    def __init__(
        self, langchain_wrapper: LangChainAPIWrapper, shared_context: ChatContext
    ):
        self.langchain_wrapper = langchain_wrapper
        self.shared_context = shared_context

    async def generate_tags(self, request: ChatGenerateTagsRequest):
        """Generate photo tags with at least one from required categories"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )

            # Set image in context if provided
            if request.image_url:
                self.shared_context.set_image(request.image_url)

            # Update context with additional information
            if hasattr(request, "cedar_state") and request.cedar_state:
                self.shared_context.set_additional_context(request.cedar_state)

            # Use context for selected tags, fallback to request
            selected_tags = (
                request.selectedTags
                or request.current_tags
                or self.shared_context.current_tags
            )
            selected_tags_str = ", ".join(selected_tags) if selected_tags else "none"

            # Build context information from shared state
            context_info = self.shared_context.get_context_summary()
            if context_info:
                context_info = f"Context: {context_info}\n"

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

            # Use shared context image URL if available
            image_url = self.shared_context.get_image_url() or request.image_url
            if not image_url:
                raise HTTPException(
                    status_code=400,
                    detail="No image URL available in context or request",
                )

            result = await self.langchain_wrapper.call_openai_vision(
                prompt=prompt,
                image_url=image_url,
                temperature=0.7,
                max_tokens=200,
            )

            tags = self._parse_tags_response(result.get("content", ""))
            tags = self._ensure_required_tags(tags)

            # Update shared context with generated tags
            all_tags = list(set(selected_tags + tags))
            self.shared_context.update_tags(all_tags)

            # Add to conversation history
            self.shared_context.add_to_conversation(
                "generate_tags", {"selected_tags": selected_tags_str}, {"tags": tags}
            )

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
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )

            # Update context if additional information provided
            if hasattr(request, "cedar_state") and request.cedar_state:
                self.shared_context.set_additional_context(request.cedar_state)

            # Use context tags, fallback to request
            current_tags = (
                request.current_tags
                or request.currentTags
                or self.shared_context.current_tags
            )
            current_tags_str = ", ".join(current_tags)
            max_tags = request.max_tags or request.maxTags
            remaining_slots = max(0, max_tags - len(current_tags))

            if remaining_slots == 0:
                return JSONResponse(content={"tags": []})

            # Build context information from shared state
            context_info = self.shared_context.get_context_summary()
            if context_info:
                context_info = f"Context: {context_info}\n"

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

            # Update shared context with all tags
            all_tags = list(set(current_tags + tags))
            self.shared_context.update_tags(all_tags)

            # Add to conversation history
            self.shared_context.add_to_conversation(
                "fill_tags",
                {"current_tags": current_tags_str, "max_tags": max_tags},
                {"suggested_tags": tags},
            )

            return JSONResponse(content={"tags": tags})

        except Exception as e:
            logger.error(f"Fill tags error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to fill tags: {str(e)}"
            )

    async def generate_caption(self, request: ChatGenerateCaptionRequest):
        """Generate a caption for the photo"""
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )

            # Set image in context if provided
            if request.image_url:
                self.shared_context.set_image(request.image_url)

            # Update context if additional information provided
            if hasattr(request, "cedar_state") and request.cedar_state:
                self.shared_context.set_additional_context(request.cedar_state)

            # Use context tags, fallback to request
            tags = (
                request.tags or request.currentTags or self.shared_context.current_tags
            )
            tags_str = ", ".join(tags) if tags else "none"

            filename_info = f"Filename: {request.filename}" if request.filename else ""

            # Build context information from shared state
            context_info = self.shared_context.get_context_summary()
            if context_info:
                context_info = f"Context: {context_info}\n"

            prompt = f"""
You are an AI assistant helping to generate engaging captions for photos.

Current tags: {tags_str}
{filename_info}
{context_info}

Looking at this image, generate a captivating and descriptive caption that:
- Captures the mood or atmosphere
- Is a plausible Instagram-style caption
- Is engaging and social media friendly
- Is max 50 characters long
- Uses social media style language, not formal
- Do not use hashtags

Generate just the caption text, nothing else.
"""

            # Use shared context image URL if available
            image_url = self.shared_context.get_image_url() or request.image_url
            if not image_url:
                raise HTTPException(
                    status_code=400,
                    detail="No image URL available in context or request",
                )

            result = await self.langchain_wrapper.call_openai_vision(
                prompt=prompt,
                image_url=image_url,
                temperature=0.8,
                max_tokens=100,
            )

            caption = self._clean_caption(result.get("content", "").strip())

            # Update shared context with generated caption
            self.shared_context.update_caption(caption)

            # Add to conversation history
            self.shared_context.add_to_conversation(
                "generate_caption",
                {"tags": tags_str, "filename": request.filename or ""},
                {"caption": caption},
            )

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
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )

            # Set image in context if provided
            if request.image_url:
                self.shared_context.set_image(request.image_url)

            # Update context if additional information provided
            if hasattr(request, "cedar_state") and request.cedar_state:
                self.shared_context.set_additional_context(request.cedar_state)

            # Use context tags, fallback to request
            tags = request.tags or self.shared_context.current_tags
            tags_str = ", ".join(tags) if tags else "none"

            # Use current caption from request or context
            current_caption = (
                request.current_caption
                or request.currentCaption
                or self.shared_context.current_caption
            )

            if not current_caption:
                raise HTTPException(
                    status_code=400, detail="No current caption provided"
                )

            # Build context information from shared state
            context_info = self.shared_context.get_context_summary()
            if context_info:
                context_info = f"Context: {context_info}\n"

            prompt = f"""
You are an AI assistant helping to enhance photo captions.

Current caption: "{current_caption}"
Tags: {tags_str}
{context_info}

Please FINISH this caption, DO NOT OVERWRITE WHAT IS ALREADY THERE. Start from the existing caption and make it more engaging. The enhanced caption should:
- Captures the mood or atmosphere
- Is a plausible Instagram-style caption
- Is engaging and social media friendly
- Is max 50 characters long
- Uses social media style language, not formal
- Do not use hashtags

Return just the enhanced caption text, nothing else.
"""

            # Use shared context image URL or request image URL
            image_url = self.shared_context.get_image_url() or request.image_url

            if image_url:
                result = await self.langchain_wrapper.call_openai_vision(
                    prompt=prompt,
                    image_url=image_url,
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

            # Update shared context with enhanced caption
            self.shared_context.update_caption(caption)

            # Add to conversation history
            self.shared_context.add_to_conversation(
                "fill_caption",
                {"original_caption": current_caption, "tags": tags_str},
                {"enhanced_caption": caption},
            )

            return JSONResponse(content={"caption": caption})

        except Exception as e:
            logger.error(f"Fill caption error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to enhance caption: {str(e)}"
            )

    async def filter_images(self, request: ChatFilterImagesRequest):
        """Generate backend filter objects based on cedar state and frontend activeFilters.

        The method uses the same LLM wrapper as other handlers to generate a JSON list of
        filters suitable for applying to image queries. Falls back to simple equality
        filters constructed from the activeFilters if parsing/LLM fails.
        """
        try:
            if not self.langchain_wrapper.openai_available:
                raise HTTPException(
                    status_code=500, detail="OpenAI client not available"
                )

            # Normalize fields that may come in camelCase or snake_case
            cedar_state = (
                getattr(request, "cedar_state", None)
                or getattr(request, "cedarState", None)
                or {}
            )
            active = (
                getattr(request, "active_filters", None)
                or getattr(request, "activeFilters", None)
                or (cedar_state or {}).get("activeFilters")
                or {}
            )
            available = (
                getattr(request, "available_filters", None)
                or getattr(request, "availableFilters", None)
                or {}
            )
            all_tags = (
                getattr(request, "all_tags", None)
                or getattr(request, "allTags", None)
                or []
            )
            all_user_ids = (
                getattr(request, "all_user_ids", None)
                or getattr(request, "allUserIds", None)
                or []
            )
            limit = getattr(request, "limit", None) or 20

            # Build a concise prompt describing available options and the desired output
            prompt = (
                "You are an assistant that converts a frontend 'cedar' chat state and user-selected "
                "filter inputs into a backend-ready list of filter objects.\n\n"
            )
            prompt += f"Available filters/options: {json.dumps(available)}\n"
            prompt += f"All tags: {json.dumps(all_tags)}\n"
            prompt += f"All userIds: {json.dumps(all_user_ids)}\n"
            prompt += f"Cedar state summary (messages and currentThreadId): {json.dumps({'messages': cedar_state.get('messages') if isinstance(cedar_state, dict) else None, 'currentThreadId': cedar_state.get('currentThreadId') if isinstance(cedar_state, dict) else None})}\n"
            prompt += f"Active filters requested by user: {json.dumps(active)}\n\n"
            prompt += (
                "Return only JSON. The top-level object must contain a key 'filters' whose value is a list of "
                'filter objects. Each filter object should be {"field": <field_name>, "operator": <op>, "value": <value>}.'
            )
            prompt += f"\nInclude at most {limit} filters and only include fields relevant to available filters.\n"

            messages = [{"role": "user", "content": prompt}]

            # Ask the LLM to produce structured filters
            # Include natural language filter if provided
            nl_filter = getattr(request, "natural_language_filter", None) or getattr(
                request, "naturalLanguageFilter", None
            )
            if nl_filter:
                messages.append(
                    {
                        "role": "user",
                        "content": f"User's natural language filter: {nl_filter}",
                    }
                )

            result = await self.langchain_wrapper.call_openai_chat(
                messages=messages, temperature=0.0, max_tokens=500
            )

            content = result.get("content", "")

            # Helper to extract JSON from text
            def _extract_json(text: str):
                text = text.strip()
                # Try to find the first JSON object or array and decode
                first_obj = text.find("{")
                first_arr = text.find("[")
                try:
                    if first_obj != -1:
                        return json.loads(text[first_obj:])
                    if first_arr != -1:
                        return json.loads(text[first_arr:])
                    # As a last resort, try to parse whole text
                    return json.loads(text)
                except Exception:
                    return None

            parsed = _extract_json(content)

            # If parsing failed, ask model once to reformat its previous answer into JSON only
            if parsed is None:
                followup_prompt = (
                    "The previous response could not be parsed as valid JSON.\n"
                    "Please extract or convert the response into a single JSON object with a top-level key 'filters' whose value is a list of filter objects. "
                    'Each filter object must be of the form {"field": <field_name>, "operator": <op>, "value": <value>}. '
                    "Do NOT include any additional text. Return ONLY the JSON object.\n\n"
                    f"Previous model output: {content}"
                )

                reform_messages = [{"role": "user", "content": followup_prompt}]

                reform_result = await self.langchain_wrapper.call_openai_chat(
                    messages=reform_messages, temperature=0.0, max_tokens=300
                )

                reform_text = reform_result.get("content", "")
                parsed = _extract_json(reform_text)

            if parsed is None:
                # Give up after one retry — prefer failing loudly rather than silently applying heuristics
                logger.error(
                    "LLM did not return valid JSON for filters. Raw output: %s", content
                )
                raise HTTPException(
                    status_code=500,
                    detail="OpenAI did not return valid JSON for filters. Check model response logs.",
                )

            # parsed may be an object with 'filters' key or a list
            if isinstance(parsed, dict) and "filters" in parsed:
                filters = parsed.get("filters", [])
            elif isinstance(parsed, list):
                filters = parsed
            else:
                # If parsed is an object but not containing 'filters', try to coerce
                filters = parsed.get("filters") if isinstance(parsed, dict) else []

            # Limit filters
            if isinstance(filters, list) and len(filters) > limit:
                filters = filters[:limit]

            # Save to conversation history for traceability
            try:
                self.shared_context.add_to_conversation(
                    "filter_images",
                    {"active_filters": active, "available": available},
                    {"filters": filters},
                )
            except Exception:
                # Non-fatal if shared context fails
                logger.debug("Failed to add filter_images to conversation history")

            # Persist active/derived filters into shared context for frontend state sync
            try:
                # store full filter payload in additional_context under 'filters'
                self.shared_context.set_additional_context(
                    {
                        "activeFilters": active,
                        "availableFilters": available,
                        "generatedFilters": filters,
                        "allTags": all_tags,
                        "allUserIds": all_user_ids,
                    }
                )

                # If filters include tag constraints, update current_tags in context
                tag_values = []
                for f in filters:
                    try:
                        field = f.get("field", "").lower()
                        val = f.get("value")
                        if field in ("tag", "tags"):
                            if isinstance(val, list):
                                tag_values.extend(val)
                            else:
                                tag_values.append(val)
                    except Exception:
                        continue

                if tag_values:
                    # dedupe and set
                    unique_tags = list(dict.fromkeys([t for t in tag_values if t]))
                    self.shared_context.update_tags(unique_tags)
            except Exception:
                logger.debug("Failed to persist filters into shared context")

            return JSONResponse(content={"filters": filters}, status_code=200)

        except Exception as e:
            logger.error(f"Filter images error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate filters: {str(e)}"
            )

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
