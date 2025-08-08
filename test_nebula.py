import os
import pathlib
import logging
from eventrag import EventRAG, QueryParam
from eventrag.llm import gpt_4o_complete, openai_complete, openai_embedding
from dotenv import load_dotenv
from eventrag.utils import wrap_embedding_func_with_attrs
# import weave
from eventrag.kg.milvus_impl import MilvusVectorDBStorge
from eventrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from eventrag.utils import EmbeddingFunc

load_dotenv(override=True)
# weave.init("EventRAG")

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

# WORKING_DIR = "./fraud_data/administrative_cases"
# WORKING_DIR = "./fraud_data/civil_cases"
WORKING_DIR = "./fraud_data/compensation_cases"
# WORKING_DIR = "./fraud_data/criminal_cases"
# WORKING_DIR = "./fraud_data/enforcement_cases"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Qwen embedding configuration
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=512)
async def jina_embedding(texts, **kwargs):
    """Qwen embedding wrapper using OpenAI-compatible interface"""
    return await openai_embedding(
        texts=texts,
            base_url="http://10.10.202.242:2097/v1", # automatically use the base URL from environment variables
        # api_key="sk-1",  # automatically use the API key from environment variables
        model="bce-embedding-base_v1",  # Qwen's text embedding model
        dimensions=768,  # Specify dimensions for Qwen embedding
    )

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,  # Use OpenAI API, automatically use API key from environment variables
    graph_storage="NebulaGraphStorage",
    vector_storage="NanoVectorDBStorage",
    embedding_func=jina_embedding,  # Use Qwen text embedding
    embedding_batch_num=10,
    llm_model_name="Qwen3-32B",  # Use qwen-plus model
    # llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    # chunk_token_size=50,
    # chunk_overlap_token_size=10,
    node2vec_params= {
            "dimensions": 768,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
)


import time
# 记录开始时间
start_time = time.time()

failed_ls = []
logger = logging.getLogger("eventrag-bioplanner")
for file in pathlib.Path(WORKING_DIR).glob("*.txt"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            rag.insert(f.read())
        logger.info(f"🎇Inserted {file}")
    except Exception as e:
        failed_ls.append(file)
        logger.error(f"💥Failed to insert {file}: {e}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"建图执行时间: {elapsed_time:.2f} 秒")
logger.info(f"💥Failed to insert {len(failed_ls)} files, {failed_ls}")

# with open("chat_history_neo4j/chatlog_processed.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

# Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

start_time = time.time()
# Perform multi-event reasoning
print(
    rag.query("人物关系是怎么样的？", param=QueryParam(mode="agent"))
)
# 计算并打印执行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"查询执行时间: {elapsed_time:.2f} 秒")