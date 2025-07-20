import os
from eventrag import EventRAG, QueryParam
from eventrag.llm import openai_complete_if_cache, openai_embedding
from eventrag.utils import wrap_embedding_func_with_attrs
import numpy as np
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# DeepSeek API configuration
async def deepseek_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    """DeepSeek API wrapper using OpenAI-compatible interface"""
    return await openai_complete_if_cache(
        model="deepseek-chat",  # DeepSeek's chat model
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
        api_key=os.getenv("DEEPSEEK_API_KEY") or "sk-",  # Replace with your actual API key
        **kwargs
    )

# Qwen embedding configuration
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def qwen_embedding(texts, **kwargs):
    """Qwen embedding wrapper using OpenAI-compatible interface"""
    return await openai_embedding(
        texts=texts,
        model="text-embedding-v4",  # Qwen's text embedding model
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # Qwen API endpoint
        api_key=os.getenv("QWEN_API_KEY") or "sk-",  # Replace with your actual API key
        dimensions=1536,  # Specify dimensions for Qwen embedding
    )
    
# Qwen-3 API configuration
async def qwen3_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Qwen-3 API wrapper using OpenAI-compatible interface"""
    return await openai_complete_if_cache(
        model="qwen3-32b",  # Qwen-3's chat model
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Qwen-3 API endpoint
        api_key=os.getenv("QWEN_API_KEY") or "sk-",  # Replace with your actual API key
        **kwargs
    )

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=qwen3_complete,  # Use DeepSeek API
    embedding_func=qwen_embedding,  # Use Qwen text embedding
    embedding_batch_num=10
    # llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./dickens/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform multi-event reasoning
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="agent"))
)
