#!/usr/bin/env python3
"""
Example usage of the logger module with the LangChain wrapper.

This demonstrates how to use the centralized logging system and configure it
on or off through environment variables or config.
"""

import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain.logger import get_logger, set_log_level, is_logging_enabled
from langchain.config import CONFIG


def main():
    """Demonstrate logger module usage"""

    # Create a logger for this script
    logger = get_logger(__name__)

    print("=== Logger Module Demo ===\n")

    # Show current logging configuration
    logging_config = CONFIG.get("logging", {})
    print(f"Logging enabled: {is_logging_enabled()}")
    print(f"Log level: {logging_config.get('level', 'INFO')}")
    print(f"Console output: {logging_config.get('console_output', True)}")
    print(f"Use colors: {logging_config.get('use_colors', True)}")
    print(f"Log file: {logging_config.get('file_path', 'None')}")
    print()

    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    print()

    # Demonstrate dynamic level changes
    print("Changing log level to DEBUG...")
    set_log_level("DEBUG")
    logger.debug("This DEBUG message should now be visible")
    print()

    print("Changing log level back to INFO...")
    set_log_level("INFO")
    logger.debug("This DEBUG message should not be visible")
    logger.info("This INFO message should be visible")
    print()

    # Test with OpenAI wrapper (stub since no API key)
    try:
        from langchain.wrapper import LangChainAPIWrapper

        logger.info("Creating LangChainAPIWrapper instance...")
        wrapper = LangChainAPIWrapper()
        logger.info(
            f"Wrapper created successfully! OpenAI available: {wrapper.openai_available}"
        )

    except Exception as e:
        logger.error(f"Failed to create wrapper: {e}")

    print("\n=== Environment Variables for Configuration ===")
    print("You can configure logging using these environment variables:")
    print("- LANGCHAIN_LOGGING_ENABLED=true/false")
    print("- LANGCHAIN_LOG_LEVEL=DEBUG/INFO/WARNING/ERROR/CRITICAL")
    print("- LANGCHAIN_LOG_FILE=/path/to/logfile.log")
    print("- LANGCHAIN_LOG_COLORS=true/false")
    print()
    print("Example usage:")
    print("LANGCHAIN_LOGGING_ENABLED=false python examples.py")
    print("LANGCHAIN_LOG_LEVEL=DEBUG python examples.py")


if __name__ == "__main__":
    main()
