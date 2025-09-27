"""
Filter Generation API Set

Specialized API set for generating filter configurations from natural language.
This translates natural language queries into structured filter configurations.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .models import (
    APIResponse,
    FilterGenerationRequest,
    FilterGenerationResponse,
    logger,
)
from .wrapper import LangChainAPIWrapper


class FilterGenerationAPISet:
    """
    Specialized API set for generating filter configurations from natural language.

    This translates natural language queries into structured filter configurations.
    """

    def __init__(self, wrapper: LangChainAPIWrapper):
        self.wrapper = wrapper

    async def generate_filters(self, request: FilterGenerationRequest) -> APIResponse:
        """Generate filter configuration from natural language"""

        # Create the chain for filter generation
        chain = self.wrapper.create_simple_chain(self._process_filter_generation)

        return await self.wrapper.execute_chain(chain, request.dict())

    async def validate_filters(
        self, filters: List[Dict[str, Any]], available_filters: List[Dict[str, Any]]
    ) -> APIResponse:
        """Validate generated filters against available filter definitions"""

        chain = self.wrapper.create_simple_chain(self._process_filter_validation)

        inputs = {"filters": filters, "available_filters": available_filters}

        return await self.wrapper.execute_chain(chain, inputs)

    async def _process_filter_generation(
        self, inputs: Dict[str, Any]
    ) -> FilterGenerationResponse:
        """Process natural language to generate filters using OpenAI"""

        query = inputs.get("natural_language_query", "")
        available_filters = inputs.get("available_filters", [])
        max_filters = inputs.get("max_filters", 5)

        # Try to use OpenAI Chat API if available
        assert self.wrapper is not None, "API wrapper is not initialized"

        try:
            prompt = self._build_filter_generation_prompt(
                query, available_filters, max_filters
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at converting natural language queries into structured database filters.",
                },
                {"role": "user", "content": prompt},
            ]

            openai_response = await self.wrapper.call_openai_chat(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=1000,
            )

            # Parse OpenAI response into structured filters
            filters_data = self._parse_openai_filter_response(
                openai_response["content"], available_filters, max_filters
            )

            return FilterGenerationResponse(
                generated_filters=filters_data["filters"],
                explanation=filters_data["explanation"],
                confidence_score=filters_data["confidence"],
                alternative_suggestions=filters_data.get("alternatives", []),
            )

        except Exception as e:
            logger.warning(f"OpenAI Chat API failed: {str(e)}")
            return FilterGenerationResponse(
                generated_filters=[],
                explanation="Failed to generate filters due to an error.",
                confidence_score=0.0,
                alternative_suggestions=[],
            )

    async def _process_filter_validation(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate filters against available filter definitions"""

        filters = inputs.get("filters", [])
        available_filters = inputs.get("available_filters", [])

        # Simulate validation logic
        await asyncio.sleep(0.1)

        valid_filters = []
        invalid_filters = []

        available_fields = {f.get("field") for f in available_filters}

        for filter_config in filters:
            if filter_config.get("field") in available_fields:
                valid_filters.append(filter_config)
            else:
                invalid_filters.append(filter_config)

        return {
            "valid_filters": valid_filters,
            "invalid_filters": invalid_filters,
            "validation_summary": {
                "total_filters": len(filters),
                "valid_count": len(valid_filters),
                "invalid_count": len(invalid_filters),
            },
        }

    def _build_filter_generation_prompt(
        self, query: str, available_filters: List[Dict[str, Any]], max_filters: int
    ) -> str:
        """Build a prompt for OpenAI to generate filters from natural language"""

        # Build description of available filters
        filter_descriptions = []
        for filter_def in available_filters:
            field = filter_def.get("field", "unknown")
            filter_type = filter_def.get("type", "unknown")
            operators = filter_def.get("operators", [])
            display_name = filter_def.get("display_name", field)

            desc = f"- {display_name} (field: '{field}', type: '{filter_type}')"
            if operators:
                desc += f", operators: {operators}"
            if "values" in filter_def:
                desc += f", possible values: {filter_def['values']}"

            filter_descriptions.append(desc)

        prompt = f"""Convert this natural language query into structured database filters:
Query: "{query}"

Available filter fields:
{chr(10).join(filter_descriptions)}

Generate up to {max_filters} filters that match the user's intent. Return your response as a JSON object with this structure:

{{
  "filters": [
    {{
      "type": "filter_type",
      "field": "field_name",
      "operator": "operator",
      "value": "filter_value",
      "display_name": "Human readable name"
    }}
  ],
  "explanation": "Brief explanation of how the filters match the query",
  "confidence": 0.85,
  "alternatives": [
    {{
      "type": "alternative_filter_type",
      "field": "alternative_field",
      "operator": "operator",
      "value": "value",
      "display_name": "Alternative suggestion"
    }}
  ]
}}

Guidelines:
- Only use fields that exist in the available filters list
- Use appropriate operators for each field type
- For date filters, use ISO format (YYYY-MM-DD)
- For numeric filters, use appropriate numeric values
- For categorical filters, use values from the possible values list when available
- Be specific and accurate in your filter generation
- Confidence should reflect how well the filters match the query (0.0-1.0)
"""

        return prompt

    def _parse_openai_filter_response(
        self,
        response_content: str,
        available_filters: List[Dict[str, Any]],
        max_filters: int,
    ) -> Dict[str, Any]:
        """Parse OpenAI response content into structured filter data"""
        import json
        import re

        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_data = json.loads(json_str)

                # Validate the response structure
                if "filters" in response_data and isinstance(
                    response_data["filters"], list
                ):
                    # Validate each filter against available fields
                    available_fields = {f.get("field") for f in available_filters}
                    valid_filters = []

                    for filter_config in response_data["filters"]:
                        if (
                            isinstance(filter_config, dict)
                            and filter_config.get("field") in available_fields
                        ):
                            valid_filters.append(filter_config)

                    return {
                        "filters": valid_filters[:max_filters],
                        "explanation": response_data.get(
                            "explanation",
                            "Filters generated from natural language query",
                        ),
                        "confidence": float(response_data.get("confidence", 0.7)),
                        "alternatives": response_data.get("alternatives", []),
                    }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse OpenAI filter response as JSON: {e}")

        # Fallback: extract basic filter concepts from text
        return self._extract_filters_from_text(
            response_content, available_filters, max_filters
        )

    def _extract_filters_from_text(
        self, text: str, available_filters: List[Dict[str, Any]], max_filters: int
    ) -> Dict[str, Any]:
        """Extract filter concepts from natural language text as fallback"""
        import re

        available_fields = {f.get("field"): f for f in available_filters}
        extracted_filters = []

        # Simple pattern matching for common filter concepts
        text_lower = text.lower()

        # Date-related filters
        if any(
            word in text_lower
            for word in ["recent", "new", "latest", "today", "yesterday"]
        ):
            if "created_at" in available_fields or "date" in available_fields:
                field = "created_at" if "created_at" in available_fields else "date"
                extracted_filters.append(
                    {
                        "type": "date_range",
                        "field": field,
                        "operator": ">=",
                        "value": "2024-01-01",
                        "display_name": "Recent items",
                    }
                )

        # Popularity/trending filters
        if any(word in text_lower for word in ["popular", "trending", "top", "best"]):
            if "popularity_score" in available_fields or "score" in available_fields:
                field = (
                    "popularity_score"
                    if "popularity_score" in available_fields
                    else "score"
                )
                extracted_filters.append(
                    {
                        "type": "numeric_range",
                        "field": field,
                        "operator": ">=",
                        "value": "0.8",
                        "display_name": "Popular items",
                    }
                )

        # Category filters
        categories = re.findall(
            r"\b(technology|tech|science|art|music|sports|news)\b", text_lower
        )
        if categories and "category" in available_fields:
            extracted_filters.append(
                {
                    "type": "categorical",
                    "field": "category",
                    "operator": "in",
                    "value": list(set(categories)),
                    "display_name": f"Categories: {', '.join(set(categories))}",
                }
            )

        return {
            "filters": extracted_filters[:max_filters],
            "explanation": f"Extracted filters from text analysis (fallback method)",
            "confidence": 0.5,
            "alternatives": [],
        }
