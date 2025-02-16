import csv
import time
import os
from functools import wraps
from datetime import datetime
import asyncio

def log_openai_usage(file_path="openai_usage.csv"):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            if asyncio.iscoroutinefunction(func):
                response = await func(*args, **kwargs)
            else:
                response = await asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs)
            return _log_and_return(response, start_time)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            response = func(*args, **kwargs)
            return _log_and_return(response, start_time)

        def _log_and_return(response, start_time):
            end_time = time.time()
            duration = end_time - start_time

            usage_info = _extract_usage_info(response)

            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': f"{duration:.2f}",
                **usage_info
            }

            file_exists = os.path.exists(file_path)
            with open(file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)
            
            return response

        def _extract_usage_info(response):
            if hasattr(response, 'model'):
                #  OpenAI 
                model = response.model
                usage = response.usage if hasattr(response, 'usage') else None
                
                if usage:
                    return {
                        'model': model,
                        'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(usage, 'completion_tokens', 0),
                        'total_tokens': getattr(usage, 'total_tokens', 0)
                    }

            if isinstance(response, dict):
                usage = response.get('usage', {})
                return {
                    'model': response.get('model', 'unknown'),
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

            return {
                'model': 'unknown',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
            
    return decorator

if __name__ == "__main__":
    from openai import OpenAI
    client = OpenAI()
    response = log_openai_usage()(client.chat.completions.create)(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, who are you?"}],
    )
    print(response)