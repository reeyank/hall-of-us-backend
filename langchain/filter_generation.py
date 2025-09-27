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
)
from .wrapper import LangChainAPIWrapper
from .logger import get_logger

# Create a logger for this module
logger = get_logger(__name__)


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

        # Use OpenAI Chat API
        assert self.wrapper is not None, "API wrapper is not initialized"

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

        # Extract JSON from the response
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        json_str = json_match.group()
        response_data = json.loads(json_str)

        # Validate the response structure
        if "filters" in response_data and isinstance(response_data["filters"], list):
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
