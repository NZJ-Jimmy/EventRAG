from dataclasses import field
import os
from eventrag import EventRAG, QueryParam
from eventrag.llm import openai_complete_if_cache, openai_embedding, openai_complete, jina_embedding
from eventrag.utils import wrap_embedding_func_with_attrs
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./surfilter"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Qwen embedding configuration
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=512)
async def jina_embedding(texts, **kwargs):
    """Qwen embedding wrapper using OpenAI-compatible interface"""
    return await openai_embedding(
        texts=texts,
        # base_url="http://10.10.202.242:2097/v1", # automatically use the base URL from environment variables
        # api_key="sk-1",  # automatically use the API key from environment variables
        model="bce-embedding-base_v1",  # Qwen's text embedding model
        dimensions=768,  # Specify dimensions for Qwen embedding
    )

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,  # Use OpenAI API, automatically use API key from environment variables
    embedding_func=jina_embedding,  # Use Qwen text embedding
    embedding_batch_num=10,
    llm_model_name="qwen-plus",  # Use qwen-plus model
    # llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    node2vec_params= {
            "dimensions": 768,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
)

with open("./surfilter/desc.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
# print(
#     rag.query("任子行是什么时候成立的？", param=QueryParam(mode="naive"))
# )

# Perform multi-event reasoning
print(
    rag.query("任子行是什么时候成立的？", param=QueryParam(mode="agent"))
)
