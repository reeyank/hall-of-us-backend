"""
LangChain Wrapper Component for API Calls

This module provides a centralized wrapper for all API interactions using LangChain.
It handles common concerns like error handling, retries, logging, and response formatting.
Now includes OpenAI integration for LLM and Vision API calls.
"""

import sys
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import asyncio

from .models import APIResponse
from .logger import get_logger

# Create a logger for this module
logger = get_logger(__name__)

# OpenAI integration
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")
    sys.exit(1)


try:
    # Try the newer LangChain imports first
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.runnables import Runnable, RunnableLambda
except ImportError:
    try:
        logger.warning("Failed to import from langchain_core.")
        # Fall back to older LangChain imports
        from langchain.callbacks.base import AsyncCallbackHandler
        from langchain.schema.runnable import Runnable, RunnableLambda
    except ImportError:
        # If LangChain is not available, create stub classes
        logger.warning("LangChain not available, using stub implementations")

        sys.exit(1)

import langchain

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


class APICallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for monitoring API calls"""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.errors = []

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        self.start_time = datetime.now()
        logger.info(f"Starting API chain with inputs: {inputs}")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if self.start_time:
            execution_time = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.info(f"API chain completed in {execution_time:.2f}ms")

    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"API chain error: {str(error)}")
        self.errors.append(str(error))


class LangChainAPIWrapper:
    """
    Main LangChain component that wraps API calls with common functionality.

    This wrapper provides:
    - Standardized response format
    - Error handling and retries
    - Logging and monitoring
    - Rate limiting (can be extended)
    - Caching (can be extended)
    - OpenAI integration for LLM and Vision API calls
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        openai_api_key: Optional[str] = None,
    ):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.callback_handler = APICallbackHandler()

        # Initialize OpenAI client
        load_dotenv()  # Load environment variables from .env file
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            self.openai_client = AsyncOpenAI(api_key=api_key)
            self.openai_available = True
            logger.info("OpenAI client initialized successfully")
        else:
            self.openai_client = None
            self.openai_available = False
            if not api_key:
                logger.warning(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
                )
            else:
                logger.warning("OpenAI not available. Install with: pip install openai")

    async def call_openai_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a chat completion call to OpenAI API

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters

        Returns:
            Dict containing the response and metadata
        """
        if not self.openai_available:
            raise ValueError(
                "OpenAI client not available. Check API key and installation."
            )

        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

            if stream:
                return response  # Return the streaming response directly
            else:
                return {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }

        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {str(e)}")
            raise e

    async def call_openai_vision(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a vision API call to OpenAI for image analysis

        Args:
            prompt: Text prompt describing what to analyze
            image_url: URL of the image to analyze
            model: OpenAI vision model to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters

        Returns:
            Dict containing the response and metadata
        """
        if not self.openai_available:
            raise ValueError(
                "OpenAI Vision client not available. Check API key and installation."
            )

        if not image_url:
            raise ValueError("Either image_url must be provided")

        # Prepare the image content
        image_content = {"type": "image_url", "image_url": {"url": image_url}}

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, image_content],
            }
        ]

        try:
            response = await self.openai_client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, **kwargs
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }

        except Exception as e:
            logger.error(f"OpenAI vision API call failed: {str(e)}")
            raise e

    async def execute_chain(
        self, chain: Runnable, inputs: Dict[str, Any], retry_count: int = 0
    ) -> APIResponse:
        """
        Execute a LangChain runnable with error handling and retries.

        Args:
            chain: The LangChain runnable to execute
            inputs: Input parameters for the chain
            retry_count: Current retry attempt (used internally)

        Returns:
            APIResponse: Standardized response object
        """
        start_time = datetime.now()

        try:
            # Execute the chain with timeout
            result = await asyncio.wait_for(
                chain.ainvoke(inputs, config={"callbacks": [self.callback_handler]}),
                timeout=self.timeout_seconds,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return APIResponse(
                success=True,
                data=result,
                timestamp=datetime.now(),
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError as e:
            logger.error(f"API call timed out after {self.timeout_seconds} seconds")
            return await self._handle_error(e, chain, inputs, retry_count, start_time)

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return await self._handle_error(e, chain, inputs, retry_count, start_time)

    async def _handle_error(
        self,
        error: Exception,
        chain: Runnable,
        inputs: Dict[str, Any],
        retry_count: int,
        start_time: datetime,
    ) -> APIResponse:
        """Handle errors with retry logic"""

        if retry_count < self.max_retries:
            logger.info(
                f"Retrying API call (attempt {retry_count + 1}/{self.max_retries})"
            )
            await asyncio.sleep(2**retry_count)  # Exponential backoff
            return await self.execute_chain(chain, inputs, retry_count + 1)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return APIResponse(
            success=False,
            error=str(error),
            timestamp=datetime.now(),
            execution_time_ms=execution_time,
        )

    async def batch_execute(
        self, chains_and_inputs: List[tuple[Runnable, Dict[str, Any]]]
    ) -> List[APIResponse]:
        """
        Execute multiple chains in parallel.

        Args:
            chains_and_inputs: List of (chain, inputs) tuples

        Returns:
            List of APIResponse objects
        """
        tasks = [
            self.execute_chain(chain, inputs) for chain, inputs in chains_and_inputs
        ]

        return await asyncio.gather(*tasks)

    def create_simple_chain(self, api_function: Callable) -> Runnable:
        """
        Create a simple LangChain runnable from a regular async function.

        Args:
            api_function: Async function to wrap

        Returns:
            Runnable that can be executed by the wrapper
        """
        return RunnableLambda(api_function)


# Global instance
api_wrapper = LangChainAPIWrapper()
