import re
import json
import os
from openai import OpenAI, AsyncOpenAI
from time import sleep
from rich_show import progress
import argparse
import asyncio


async def evaluate_single_pair(client, query, answer1, answer2, i):
    sys_prompt = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, **Directness** and **Logic**.
"""

    prompt = f"""
You will evaluate two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, **Directness** and **Logic**.
- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Directness**: How specifically and clearly does the answer address the question?
- **Logic**: How logical and coherent is the answer in its structure and reasoning?
For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these 5 categories.
Here is the question:
{query}
Here are the two answers:
**Answer 1:**
{answer1}
**Answer 2:**
{answer2}
Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.
Output your evaluation in the following JSON format:
{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the five criteria]"
    }}
}}
"""

    error_count = 0
    while error_count < 3:
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "question_id": i + 1,
                "evaluation": response.choices[0].message.content
            }
        except Exception as e:
            error_count += 1
            print(f"Error: {e}")
            await asyncio.sleep(3)
    return None


async def process_batch(client, batch_data):
    tasks = []
    for query, answer1, answer2, i in batch_data:
        task = evaluate_single_pair(client, query, answer1, answer2, i)
        tasks.append(task)
    return await asyncio.gather(*tasks)


async def realtime_eval(result1_file, result2_file, output_file):
    client = AsyncOpenAI()

    queries = []

    with open(result1_file, "r") as f:
        answers1_raw = json.load(f)

    answers1_dict = {}
    for item in answers1_raw:
        query = item["query"]
        result = item["result"]
        if query not in answers1_dict or len(result) > len(answers1_dict[query]["result"]):
            answers1_dict[query] = item

    with open(result2_file, "r") as f:
        answers2_raw = json.load(f)

    answers2_dict = {}
    for item in answers2_raw:
        query = item["query"]
        result = item["result"]
        if query not in answers2_dict or len(result) > len(answers2_dict[query]["result"]):
            answers2_dict[query] = item

    common_queries = set(answers1_dict.keys()) & set(answers2_dict.keys())

    queries = list(common_queries)

    answers1 = [answers1_dict[q]["result"] for q in queries]
    answers2 = [answers2_dict[q]["result"] for q in queries]
    print(len(answers1), len(answers2))

    results = []
    batch_size = 10

    with progress:
        eval_task = progress.add_task("Processing evaluation", total=len(queries))

        for i in range(0, len(queries), batch_size):
            batch_data = [
                (query, answer1, answer2, idx)
                for idx, (query, answer1, answer2)
                in enumerate(list(zip(queries, answers1, answers2))[i:i + batch_size], i)
            ]

            batch_results = await process_batch(client, batch_data)
            results.extend([r for r in batch_results if r is not None])
            progress.update(eval_task, advance=len(batch_data))

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"All evaluations completed and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method1", type=str, required=True)
    parser.add_argument("--method2", type=str, required=True)
    args = parser.parse_args()
    WORK_DIR = f"../RAG-Data"
    RESULT_DIR = f"{WORK_DIR}/Evalution"
    print(f"Evaluating {args.dataset} {args.method1} VS {args.method2}")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    asyncio.run(realtime_eval(
        result1_file=f"{WORK_DIR}/{args.method1}/{args.dataset}/result.json",
        result2_file=f"{WORK_DIR}/{args.method2}/{args.dataset}/result.json",
        output_file=f"{RESULT_DIR}/{args.dataset}-{args.method1}-VS-{args.method2}.json",
    ))
