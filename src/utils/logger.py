import logging
from config import settings


def setup_logger(name: str = "intent-matcher") -> logging.Logger:

    # Get or create a logger with this name
    # If a logger with this name already exists, we'll get the existing one
    logger = logging.getLogger(name)

    # This controls which messages are actually displayed
    # DEBUG: Very detailed, for troubleshooting
    # INFO: General informational messages (default)
    # WARNING: Something unexpected but not critical
    # ERROR: Something went wrong
    # CRITICAL: Serious error, program might not continue
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Check if handlers already exist to avoid adding duplicates
    # This can happen if setup_logger() is called multiple times
    if not logger.handlers:
        # Create a handler that writes to the console (stdout)
        console_handler = logging.StreamHandler()
        
        # Create a formatter that specifies how log messages should look
        # Format: timestamp - logger_name - level - message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Apply the formatter to the handler
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()