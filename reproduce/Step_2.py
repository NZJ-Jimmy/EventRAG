# python reproduce/Step_2.py --input_dir data/my_texts --output_file results/my_questions.txt --model gpt-4 --token_limit 1500
import json
from openai import OpenAI
from transformers import GPT2Tokenizer
import argparse
import glob
import os


def openai_complete_if_cache(
    model="gpt-4o", prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_client = OpenAI()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : -1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


def main():
    parser = argparse.ArgumentParser(description='generate questions from text')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='the directory containing text files (default: data)')
    parser.add_argument('--output_file', type=str, default='datasets/questions/questions.txt',
                        help='the output file to write questions to (default: datasets/questions/questions.txt)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='the OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--token_limit', type=int, default=2000,
                        help='the maximum number of tokens to use for each dataset description (default: 2000)')
    
    args = parser.parse_args()
    

    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists. Skipping...")
        return


    unique_contexts = []
    input_path = args.input_dir
    
    if os.path.isfile(input_path):

        with open(input_path, 'r') as f:
            unique_contexts.append(f.read())
    elif os.path.isdir(input_path):

        txt_files = glob.glob(os.path.join(input_path, "*.txt"))
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                unique_contexts.append(f.read())
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    summaries = [get_summary(context, args.token_limit) for context in unique_contexts]
    total_description = "\n\n".join(summaries)
    
    prompt = f"""
    Given the following description of a dataset:
    {total_description}
    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.
    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    result = openai_complete_if_cache(model=args.model, prompt=prompt)
    

    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as file:
        file.write(result)
    print(f"questions written to {args.output_file}")

if __name__ == "__main__":
    main()