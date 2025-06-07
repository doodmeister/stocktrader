"""
Safe Requests Utility
----------------------
Provides functionality for executing functions with retry logic and error handling.
"""

import time
from typing import Callable, Optional, TypeVar
from utils.logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar('T')  # Generic type for return values

def safe_request(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: int = 1,
    timeout: Optional[int] = None,
    error_handler: Optional[Callable[[Exception], T]] = None
) -> Optional[T]:
    """
    Execute a function with retry logic and error handling.

    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries
        timeout: Maximum seconds to wait before giving up
        error_handler: Optional function to handle exceptions

    Returns:
        The result of the function call, or None if all attempts fail
    """
    start_time = time.time()
    attempts = 0
    last_error = None

    while attempts < max_retries:
        try:
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Request timed out after {timeout} seconds")
                break

            # Execute the function
            result = func()
            return result

        except Exception as e:
            last_error = e
            attempts += 1

            if attempts < max_retries:
                wait_time = retry_delay * (2 ** (attempts - 1))  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempts}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed after {max_retries} attempts: {str(e)}")

    # All attempts failed
    if error_handler and last_error:
        return error_handler(last_error)

    return None
