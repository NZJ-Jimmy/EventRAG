
## Installation
This project uses uv as the package manager, after cloning the repository and installing uv, run the following command:
```bash
cd EventRAG
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

## Quick Start


Use the below Python snippet (in a script) to initialize EventRAG and perform queries:

```python
import os
from eventrag import EventRAG, QueryParam
from eventrag.llm import gpt_4o_mini_complete, gpt_4o_complete

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./book.txt") as f:
    rag.insert(f.read())

# Perform multi-event reasoning
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="agent")))

```

## Reproduce
The code can be found in the `./reproduce` directory.

In all the experiments, we use the `gpt_4o_complete` model for LLM generation, and the `openai_embedding` function for embedding.

We use Neo4J as the graph storage and Milvus as the vector storage for all experiments. You can setup them using the dockert-compose file in the `./dockers` directory.


## EventRAG init parameters

| **Parameter** | **Type** | **Explanation** | **Default** |
| --- | --- | --- | --- |
| **working\_dir** | `str` | Directory where the cache will be stored | `eventrag_cache+timestamp` |
| **kv\_storage** | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`, `OracleKVStorage` | `JsonKVStorage` |
| **vector\_storage** | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`, `OracleVectorDBStorage` | `NanoVectorDBStorage` |
| **graph\_storage** | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`, `Neo4JStorage`, `OracleGraphStorage` | `NetworkXStorage` |
| **log\_level** |     | Log level for application runtime | `logging.DEBUG` |
| **chunk\_token\_size** | `int` | Maximum token size per chunk when splitting documents | `1200` |
| **chunk\_overlap\_token\_size** | `int` | Overlap token size between two chunks when splitting documents | `100` |
| **tiktoken\_model\_name** | `str` | Model name for the Tiktoken encoder used to calculate token numbers | `gpt-4o-mini` |
| **entity\_extract\_max\_gleaning** | `int` | Number of loops in the entity extraction process, appending history messages | `1` |
| **entity\_summary\_to\_max\_tokens** | `int` | Maximum token size for each entity summary | `500` |
| **node\_embedding\_algorithm** | `str` | Algorithm for node embedding (currently not used) | `node2vec` |
| **node2vec\_params** | `dict` | Parameters for node embedding | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding\_func** | `EmbeddingFunc` | Function to generate embedding vectors from text | `openai_embedding` |
| **embedding\_batch\_num** | `int` | Maximum batch size for embedding processes (multiple texts sent per batch) | `32` |
| **embedding\_func\_max\_async** | `int` | Maximum number of concurrent asynchronous embedding processes | `16` |
| **llm\_model\_func** | `callable` | Function for LLM generation | `gpt_4o_mini_complete` |
| **llm\_model\_name** | `str` | LLM model name for generation | `meta-llama/Llama-3.2-1B-Instruct` |
| **llm\_model\_max\_token\_size** | `int` | Maximum token size for LLM generation (affects entity relation summaries) | `32768` |
| **llm\_model\_max\_async** | `int` | Maximum number of concurrent asynchronous LLM processes | `16` |
| **llm\_model\_kwargs** | `dict` | Additional parameters for LLM generation |     |
| **vector\_db\_storage\_cls\_kwargs** | `dict` | Additional parameters for vector database (currently not used) |     |
| **enable\_llm\_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **addon\_params** | `dict` | Additional parameters, e.g., `{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"]}`: sets example limit and output language | `example_number: all examples, language: English` |
| **convert\_response\_to\_json\_func** | `callable` | Not used | `convert_response_to_json` |
| **embedding\_cache\_config** | `dict` | Configuration for question-answer caching. Contains three parameters:<br>- `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers.<br>- `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM.<br>- `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

## License
MIT License
