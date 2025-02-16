import os
import pickle
import hashlib
import functools
import asyncio
from typing import Any, Callable, TypeVar, Union, Awaitable

T = TypeVar('T')

def persistent_cache(cache_dir: str = '.pcache'):
    """Persistent cache decorator.
    
    Args:
        cache_dir: the directory to store the cache files.
    """
    
    def _get_cache_path(func_name: str, args_hash: str) -> str:
        """Get the path to the cache file"""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return os.path.join(cache_dir, f"{func_name}_{args_hash}.pkl")
        
    def _compute_args_hash(*args, **kwargs) -> str:
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(args_str.encode()).hexdigest()
        
    def decorator(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Union[T, Awaitable[T]]]:
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            cache_path = _get_cache_path(func.__name__, _compute_args_hash(*args, **kwargs))
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    print(f"Loading from cache: {cache_path}")
                    return pickle.load(f)
                    
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result
            
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            cache_path = _get_cache_path(func.__name__, _compute_args_hash(*args, **kwargs))
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    print(f"Loading from cache: {cache_path}")
                    return pickle.load(f)
                    
            result = await func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator

if __name__ == "__main__":
    @persistent_cache(cache_dir='.pcache')
    async def expensive_function(a: str, b: str) -> str:
        return a + b

    import time
    start_time = time.time()
    result = asyncio.run(expensive_function("a"*10000, "b"*10000))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # print(result)
