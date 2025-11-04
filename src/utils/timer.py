import time
from functools import wraps
from typing import Optional


def timer(func):
    # @wraps preserves the original function's name and docstring
    # Without it, func.__name__ would be "wrapper" instead of the original name
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.time()
        
        # Call the actual function
        result = func(*args, **kwargs)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Print the timing information
        print(f"⏱️  {func.__name__} took {elapsed:.4f}s")
        
        # Return the function's result
        return result
    
    return wrapper


class Timer:
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.elapsed
        print(f"⏱️  {self.name} took {elapsed:.4f}s")
        
        # Return False to allow exceptions to propagate
        # If we returned True, we would suppress any exceptions that occurred
        return False
    
    @property
    def elapsed(self) -> float:
        if self.end_time is not None:
            # Timer has finished - return total elapsed time
            return self.end_time - self.start_time
        elif self.start_time is not None:
            # Timer is still running - return time elapsed so far
            return time.time() - self.start_time
        else:
            # Timer hasn't started yet
            return 0.0