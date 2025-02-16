# python Step_3.py --dataset toy --input datasets/questions/questions.txt --output agriculture_eventrag-result.json --error agriculture_eventrag-errors.json
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import json
import asyncio
from eventrag import EventRAG, QueryParam
from time import sleep
from rich_show import progress
import dotenv
import datetime
import weave

dotenv.load_dotenv()
# weave.init("EventRAG")

def extract_queries(file_path, output_file):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)
    

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
            existing_queries = [item["query"] for item in existing_data]
            queries = [q for q in queries if q not in existing_queries]
    return queries


async def process_query(query_text, rag_instance, query_param):
    result = await rag_instance.aquery(query_text, param=query_param)
    return {"query": query_text, "result": result}, None

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


async def process_queries_batch(queries_batch, rag_instance, query_param, progress_task):
    tasks = []
    for query_text in queries_batch:
        tasks.append(process_query(query_text, rag_instance, query_param))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    progress.update(progress_task, advance=len(queries_batch))
    return results

def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()
    batch_size = 10

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        with progress:
            query_task = progress.add_task("Processing queries", total=len(queries))
            

            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                results = loop.run_until_complete(
                    process_queries_batch(batch, rag_instance, query_param, query_task)
                )
                
                for result in results:
                    if isinstance(result, tuple):
                        query_result, error = result
                        if query_result:
                            if not first_entry:
                                result_file.write(",\n")
                            json.dump(query_result, result_file, ensure_ascii=False, indent=4)
                            first_entry = False
                        if error:
                            json.dump(error, err_file, ensure_ascii=False, indent=4)
                            err_file.write("\n")
                    else: 
                        error_data = {
                            "query": batch[results.index(result)],
                            "error": str(result)
                        }
                        json.dump(error_data, err_file, ensure_ascii=False, indent=4)
                        err_file.write("\n")

        result_file.write("\n]")


def main():
    import argparse
    

    parser = argparse.ArgumentParser(description='process queries from input file')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, default='agent')
    parser.add_argument('--input', type=str, required=True)
    
    args = parser.parse_args()

    WORKING_DIR = f"../RAG-Data/EventRAG/{args.dataset}"

    rag = EventRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        log_level="INFO",
        vector_storage="MilvusVectorDBStorge",
    )

    query_param = QueryParam(mode=args.mode)

    queries = extract_queries(args.input, WORKING_DIR+'/result.json')
    run_queries_and_save_to_json(
        queries, rag, query_param, WORKING_DIR+'/result.json', WORKING_DIR+'/error.json'
    )
    print(f"🎉Processed {len(queries)} queries and saved results to {WORKING_DIR+'/result.json'} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(f"{WORKING_DIR}/query-answer-report.txt", "a") as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🎉Processed {len(queries)} queries and saved results to {WORKING_DIR+'/result.json'}\n")


def test_process_single_query(query_text, dataset="toy", mode="agent"):

    WORKING_DIR = f"../RAG-Data/EventRAG/{dataset}"


    rag = EventRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        log_level="WARNING",
        vector_storage="MilvusVectorDBStorge",
        enable_llm_cache=False,
    )


    query_param = QueryParam(mode=mode)


    loop = always_get_an_event_loop()


    result, error = loop.run_until_complete(process_query(query_text, rag, query_param))


    if result:
        return result
    else:
        return error


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Expected a result with a single record, but found multiple.")
    main()