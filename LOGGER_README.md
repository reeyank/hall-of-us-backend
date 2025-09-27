# Logger Module Documentation

The `langchain.logger` module provides a centralized, configurable logging system for the LangChain components. It can be enabled or disabled through configuration and supports various output formats and log levels.

## Features

- **Configurable on/off**: Enable or disable logging through config or environment variables
- **Color-coded console output**: Different colors for different log levels
- **File logging support**: Optional file output with configurable path
- **Environment variable overrides**: Configure logging through environment variables
- **Per-module log levels**: Set different log levels for specific modules
- **Thread-safe singleton pattern**: Ensures consistent logging configuration across the application

## Configuration

### Default Configuration

The logging system is configured through the `config.py` file:

```python
"logging": {
    "enabled": True,                    # Enable/disable logging
    "level": "INFO",                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "console_output": True,            # Output to console
    "use_colors": True,               # Color-coded output
    "file_path": "logs/langchain.log", # Log file path (None to disable)
    "logger_levels": {                # Per-module log levels
        "langchain.wrapper": "INFO",
        "langchain.image_tagging": "INFO",
        "langchain.filter_generation": "INFO",
        "openai": "WARNING",           # Suppress noisy third-party loggers
        "httpcore": "WARNING",
        "httpx": "WARNING"
    }
}
```

### Environment Variables

Override configuration using environment variables:

- `LANGCHAIN_LOGGING_ENABLED=true/false` - Enable or disable logging
- `LANGCHAIN_LOG_LEVEL=DEBUG/INFO/WARNING/ERROR/CRITICAL` - Set log level
- `LANGCHAIN_LOG_FILE=/path/to/logfile.log` - Set log file path
- `LANGCHAIN_LOG_COLORS=true/false` - Enable or disable colors

### Example Usage

```bash
# Disable logging completely
LANGCHAIN_LOGGING_ENABLED=false python main.py

# Set debug level logging
LANGCHAIN_LOG_LEVEL=DEBUG python main.py

# Log to specific file
LANGCHAIN_LOG_FILE=/var/log/langchain.log python main.py

# Disable colors (useful for log files or certain terminals)
LANGCHAIN_LOG_COLORS=false python main.py
```

## Usage in Code

### Basic Usage

```python
from langchain.logger import get_logger

# Create a logger for your module
logger = get_logger(__name__)

# Use it like any standard logger
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

### Module-specific Loggers

```python
from langchain.logger import create_module_logger

# Create a logger specifically for a module
logger = create_module_logger("my_module")
logger.info("This will show as langchain.my_module")
```

### Dynamic Configuration

```python
from langchain.logger import set_log_level, is_logging_enabled

# Check if logging is enabled
if is_logging_enabled():
    logger.info("Logging is active")

# Dynamically change log level
set_log_level("DEBUG")
logger.debug("This debug message is now visible")
```

### Integration with Existing Code

Replace existing logging setup:

```python
# Old way
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# New way
from langchain.logger import get_logger
logger = get_logger(__name__)
```

## Log Levels

The logger supports standard Python log levels:

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened, but the program is still working
- **ERROR**: Due to a more serious problem, the program couldn't perform some function
- **CRITICAL**: Very serious error occurred, program may not be able to continue

## Output Format

### Console Output (with colors)

```
[2023-12-07 10:30:15] [INFO    ] [langchain.wrapper] OpenAI client initialized successfully
[2023-12-07 10:30:16] [WARNING ] [langchain.image_tagging] OpenAI Vision API failed, falling back to stub
[2023-12-07 10:30:17] [ERROR   ] [langchain.filter_generation] OpenAI chat completion failed: API key not provided
```

### File Output (no colors)

```
[2023-12-07 10:30:15] [INFO    ] [langchain.wrapper] OpenAI client initialized successfully
[2023-12-07 10:30:16] [WARNING ] [langchain.image_tagging] OpenAI Vision API failed, falling back to stub
[2023-12-07 10:30:17] [ERROR   ] [langchain.filter_generation] OpenAI chat completion failed: API key not provided
```

## Disabling Logging

To completely disable logging for performance-critical applications:

### Method 1: Environment Variable

```bash
LANGCHAIN_LOGGING_ENABLED=false python your_script.py
```

### Method 2: Configuration

```python
# In config.py or before importing langchain modules
from langchain.config import CONFIG
CONFIG["logging"]["enabled"] = False
```

### Method 3: Runtime

```python
from langchain.logger import LoggerManager
manager = LoggerManager()
# This will disable all logging
logging.disable(logging.CRITICAL)
```

## File Logging

Logs are automatically written to the configured file path. The file and its directory structure are created automatically if they don't exist.

- Default path: `logs/langchain.log`
- UTF-8 encoding
- Automatic directory creation
- Fallback to console-only if file logging fails

## Third-party Logger Management

The logger module automatically configures commonly noisy third-party loggers to reduce spam:

- `openai`: Set to WARNING level
- `httpcore`: Set to WARNING level
- `httpx`: Set to WARNING level

Add more in the `logger_levels` configuration as needed.

## Performance Considerations

- When logging is disabled, logger calls have minimal performance impact
- File I/O is asynchronous where possible
- Log formatting is only performed when messages will actually be output
- Color formatting is only applied to console output

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check if `LANGCHAIN_LOGGING_ENABLED=false` is set
2. **Wrong log level**: Check `LANGCHAIN_LOG_LEVEL` environment variable
3. **File logging fails**: Check file permissions and disk space
4. **Colors not working**: Set `LANGCHAIN_LOG_COLORS=true` or check terminal support

### Debug Logging

To see all internal logging configuration:

```python
from langchain.logger import get_logger
from langchain.config import CONFIG
import json

logger = get_logger(__name__)
logger.info(f"Logging config: {json.dumps(CONFIG['logging'], indent=2)}")
```

## Migration from Basic Logging

To migrate existing code:

1. Replace `import logging` with `from langchain.logger import get_logger`
2. Replace `logging.getLogger(__name__)` with `get_logger(__name__)`
3. Remove any manual `basicConfig()` calls
4. Update configuration in `config.py` instead of code

The new system is backward compatible with standard Python logging practices.
