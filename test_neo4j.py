import os
import pathlib
import logging
from eventrag import EventRAG, QueryParam
from eventrag.llm import gpt_4o_complete
import dotenv   
import weave
from eventrag.kg.milvus_impl import MilvusVectorDBStorge
from eventrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from eventrag.utils import EmbeddingFunc

dotenv.load_dotenv()
weave.init("EventRAG")

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./local_neo4jWorkDir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_complete,  # Use gpt_4o_mini_complete LLM model
    graph_storage="Neo4JStorage",
    log_level="DEBUG",
        # Use Hugging Face embedding function
    # embedding_func=EmbeddingFunc(
    #     embedding_dim=384,
    #     max_token_size=5000,
    #     func=lambda texts: hf_embedding(
    #         texts,
    #         tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
    #         embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    #     )
    # ),
    vector_storage="MilvusVectorDBStorge",
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

failed_ls = []
logger = logging.getLogger("eventrag-bioplanner")
for file in pathlib.Path("./data").glob("*.txt"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            rag.insert(f.read())
        logger.info(f"🎇Inserted {file}")
    except Exception as e:
        failed_ls.append(file)
        logger.error(f"💥Failed to insert {file}: {e}")

logger.info(f"💥Failed to insert {len(failed_ls)} files, {failed_ls}")

exit()
# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform multi-event reasoning
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="agent"))
)
