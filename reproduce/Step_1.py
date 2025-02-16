import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import json
import time
import logging
from eventrag import EventRAG, QueryParam
from eventrag.llm import gpt_4o_complete
import dotenv   
import weave
import pandas as pd
from datetime import datetime

dotenv.load_dotenv()
if len(sys.argv) < 2:
    print("please input dataset name")
    sys.exit(1)
dataset = sys.argv[1]
# weave.init("EventRAG")


WORKING_DIR = f"../RAG-Data/EventRAG/{dataset}"

def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 1
    failed_ls = []
    
    # while retries < max_retries:
        # try:
        #     rag.insert(unique_contexts)
        #     logging.info(f"✅Inserted {file_path}")
        #     break
        # except Exception as e:
        #     failed_ls.append(file_path)
        #     retries += 1
        #     logging.error(f"💥Insertion failed, retrying ({retries}/{max_retries}). Error: {str(e)}")
        #     time.sleep(10**retries)
    rag.insert(unique_contexts)
    
    if retries == max_retries:
        logging.error("💥Insertion failed after exceeding the maximum number of retries")
    if failed_ls:
        failed_ls = set(failed_ls)
        logging.error(f"💥Failed to insert {len(failed_ls)} files: {failed_ls}")


if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_complete,  # Use gpt_4o_mini_complete LLM model
    graph_storage="Neo4JStorage",
    log_level="INFO",
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
)


start_time = datetime.now()

insert_text(rag, f"../RAG-Data/unique_contexts/{dataset}_unique_contexts_small.json")

end_time = datetime.now()


df = pd.read_csv("openai_usage.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

df = df[df["timestamp"] >= start_time]
df = df[df["timestamp"] <= end_time]

df_sum = df.groupby("model")[["prompt_tokens", "completion_tokens", "total_tokens"]].sum()
print(df_sum)


df.to_csv(f"{WORKING_DIR}/openai_usage_{start_time.strftime('%Y-%m-%d-%H-%M-%S')}_{end_time.strftime('%Y-%m-%d-%H-%M-%S')}.csv", index=False)
df_sum.to_csv(f"{WORKING_DIR}/openai_usage_sum_{start_time.strftime('%Y-%m-%d-%H-%M-%S')}_{end_time.strftime('%Y-%m-%d-%H-%M-%S')}.csv", index=True)
print(f"🎉Step 1: Create kb for dataset: {dataset} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")