"""
LangChain Wrapper Component for API Calls

This module provides a centralized wrapper for all API interactions using LangChain.
It handles common concerns like error handling, retries, logging, and response formatting.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import asyncio

from .models import APIResponse, logger

try:
    # Try the newer LangChain imports first
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.runnables import Runnable, RunnableLambda
except ImportError:
    try:
        # Fall back to older LangChain imports
        from langchain.callbacks.base import AsyncCallbackHandler
        from langchain.schema.runnable import Runnable, RunnableLambda
    except ImportError:
        # If LangChain is not available, create stub classes
        logger.warning("LangChain not available, using stub implementations")

        class AsyncCallbackHandler:
            """Stub callback handler when LangChain is not available"""
            def __init__(self):
                self.start_time = None
                self.errors = []

            async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
                self.start_time = datetime.now()
                logger.info(f"Starting API chain with inputs: {inputs}")

            async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
                if self.start_time:
                    execution_time = (datetime.now() - self.start_time).total_seconds() * 1000
                    logger.info(f"API chain completed in {execution_time:.2f}ms")

            async def on_chain_error(self, error: Exception, **kwargs) -> None:
                logger.error(f"API chain error: {str(error)}")
                self.errors.append(str(error))

        class Runnable:
            """Stub runnable when LangChain is not available"""
            def __init__(self, func: Callable):
                self.func = func

            async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict] = None) -> Any:
                return await self.func(inputs)

        class RunnableLambda(Runnable):
            """Stub RunnableLambda when LangChain is not available"""
            pass


class APICallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for monitoring API calls"""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.errors = []

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
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
    """

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 30):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.callback_handler = APICallbackHandler()

    async def execute_chain(
        self,
        chain: Runnable,
        inputs: Dict[str, Any],
        retry_count: int = 0
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
                timeout=self.timeout_seconds
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return APIResponse(
                success=True,
                data=result,
                timestamp=datetime.now(),
                execution_time_ms=execution_time
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
        start_time: datetime
    ) -> APIResponse:
        """Handle errors with retry logic"""

        if retry_count < self.max_retries:
            logger.info(f"Retrying API call (attempt {retry_count + 1}/{self.max_retries})")
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            return await self.execute_chain(chain, inputs, retry_count + 1)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return APIResponse(
            success=False,
            error=str(error),
            timestamp=datetime.now(),
            execution_time_ms=execution_time
        )

    async def batch_execute(
        self,
        chains_and_inputs: List[tuple[Runnable, Dict[str, Any]]]
    ) -> List[APIResponse]:
        """
        Execute multiple chains in parallel.

        Args:
            chains_and_inputs: List of (chain, inputs) tuples

        Returns:
            List of APIResponse objects
        """
        tasks = [
            self.execute_chain(chain, inputs)
            for chain, inputs in chains_and_inputs
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
