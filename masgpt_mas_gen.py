import os
import json
import sys
import argparse
import re
from openai import OpenAI
import random
import concurrent.futures
from tqdm import tqdm

from vllm import LLM, SamplingParams
from template import form_instruction, SYSTEM_PROMPT

def form_conversations(prompt_list):
    conversations = []
    for prompt in prompt_list:
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        conversations.append(conversation)
    return conversations

def call_llm(args, prompt, temperature=0.0):
    try:
        llm = OpenAI(base_url=f"{args.masgpt_serving_url}", api_key="EMPTY")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        completion = llm.chat.completions.create(
            model=f"{args.masgpt_name}",
            messages=messages,
            stop=['<|eot_id|>'],
            temperature=temperature,
            max_tokens=4096
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"Error calling LLM for prompt: {prompt[:30]}... - {str(e)}")
        return None

def get_mas_serving(args, prompts, max_workers=50):
    """
    Get the generated MAS from the serving LLM.
    """
    generated_mas_list = [None] * len(prompts)

    # use ThreadPoolExecutor for parallelization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all scoring tasks to the executor
        future_to_index = {executor.submit(call_llm, args, prompt): idx for idx, prompt in enumerate(prompts)}
        
        # wait for all tasks to complete and collect results
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Generating MAS"):
            idx = future_to_index[future]
            try:
                generated_mas = future.result()
                generated_mas_list[idx] = generated_mas
            except Exception as exc:
                print(f"Error occurred for prompt at index {idx}: {exc}")

    return generated_mas_list

def get_mas_local(llm, conversations):
    outputs = llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=4096), use_tqdm=True)
    return [output.outputs[0].text for output in outputs]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--masgpt_name", type=str, default="MAS-GPT-32B", help="This name is used mainly for naming the output file.")

    # type 1 inference (using the serving LLM)
    parser.add_argument("--masgpt_serving_url", type=str, default=None, help="example: http://aa.bb.cc.dd:port/v1")
    parser.add_argument("--max_workers", type=int, default=50)

    # type 2 inference (using the local LLM)
    parser.add_argument("--masgpt_model_path", type=str, default=None)
    parser.add_argument("--gpu_num", type=int, default=2)

    parser.add_argument("--test_dataset_names", type=str, nargs='+', default=["MATH", "GSM8K", "AQUA-RAT", "MedMCQA"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("="*50)
    print(json.dumps(vars(args), indent=4))

    if args.masgpt_serving_url is not None:
        use_serving = True
    elif args.masgpt_model_path is not None:
        use_serving = False
    else:
        raise ValueError("Please specify the inference method. (masgpt_serving_url or masgpt_model_path)")

    if not use_serving:
        llm = LLM(model=args.masgpt_model_path, tensor_parallel_size=args.gpu_num)
    
    for i, test_dataset_name in enumerate(args.test_dataset_names):
        
        print(f">> Processing {i}-th dataset: {test_dataset_name}")

        # ================== Define the output files ==================
        save_gen_path = f"./test/{test_dataset_name}/masgpt_{args.masgpt_name}_gen.json"

        test_data_path = f"test/data_pool/{test_dataset_name}.json"
        os.makedirs(os.path.dirname(save_gen_path), exist_ok=True)

        # ================== Load the test query pool ==================
        with open(test_data_path, "r") as f:
            sample_pool = json.load(f)
        print(f">> Initially: {len(sample_pool)} samples")

        # filter out processed queries
        processed_pool = []
        processed_queries = set()
        if os.path.exists(save_gen_path):
            with open(save_gen_path, "r") as f:
                processed_pool = json.load(f)
            for sample in processed_pool:
                processed_queries.add(sample["query"])
            print(f">> Found {len(processed_pool)} processed samples")
        sample_pool = [sample for sample in sample_pool if sample["query"] not in processed_queries]
        sample_pool = sample_pool[:100] if args.dry_run else sample_pool
        print(f">> After filtering: {len(sample_pool)} samples")
        if len(sample_pool) == 0:
            continue

        prompt_list = [sample["query"] for sample in sample_pool]

        if use_serving:
            outputs = get_mas_serving(args, prompt_list, args.max_workers)
        else:
            conversations = form_conversations(prompt_list)
            outputs = get_mas_local(llm, conversations)

        total, correct = 0, 0
        for i, output_str in enumerate(outputs):
            sample_pool[i]['masgpt_generated'] = output_str
            match = re.findall(r'```python(.*?)```', output_str, re.DOTALL)
            total += 1
            if match:
                sample_pool[i]["mas_str"] = match[-1].strip()  # extract the matched code
                correct += 1
                sample_pool[i]["extract_flag"] = True   # add the flag of successful extraction
            else:
                # use sc as default. This ensures that all samples could be processed
                sample_pool[i]["mas_str"] = "from utils import *\n\nclass MAS():\n    def __init__(self, model_list):\n        self.llm = LLM(model_list)\n\n    def forward(self, taskInfo):\n        \"\"\"\n        Design a multi agent system for reasoning and task solving.\n        5 chain-of-thought agents are created for diverse solutions, and a final decision-making agent is used to generate the final answer.\n        \"\"\"\n\n        # Step-by-step instruction for each chain-of-thought agent to reason and generate answer\n        cot_instruction = f\"Task: {taskInfo}\\n\\nPlease think step by step and then solve the task.\"\n\n        # Set the number of solutions to generate; using 5 for variety and diversity\n        N = 5\n        # Call the llm to generate each solution\n        cot_results = [self.llm.call_llm(cot_instruction) for _ in range(N)]  \n\n        # Get the instruction for the final decision-making agent based on all generated solutions\n        final_decision_instruction = self.get_final_decision_instruction(taskInfo, cot_results)\n\n        # Call the llm to process the final decision-making instruction and generate the final answer\n        final_decision_result = self.llm.call_llm(final_decision_instruction)\n\n        # Return the final solution\n        return final_decision_result\n\n    def get_final_decision_instruction(self, taskInfo, cot_results):\n        \"\"\"\n        Format an instruction for final decision-making based on a given task description and a list of solutions.\n\n        Parameters:\n        taskInfo (str): A description of the task that needs to be completed.\n        cot_results (list): A list containing solutions or reasoning steps for the task.\n\n        Returns:\n        str: A formatted instruction that includes the task description, each solution, and a prompt for final decision-making.\n        \"\"\"\n\n        # Initialize the instruction text with a general guideline\n        instruction = f\"Task:\\n{taskInfo}\\n\\n\"\n\n        # Append each solution from cot_results to the instruction\n        for i, result in enumerate(cot_results):\n            instruction += f\"Solution {i+1}:\\n{result}\\n\\n\"  # Number each solution for clarity\n\n        # Add the final prompt to encourage reasoning over the solutions and provide a final answer\n        instruction += \"Given all the above solutions, reason over them carefully and provide a final answer to the task.\"\n        \n        # Return the complete instruction text\n        return instruction"
                sample_pool[i]['extract_flag'] = False

        print(f"Total: {total}, Correct: {correct}. The others are replaced by SC")

        # append the samples to the processed pool
        processed_pool.extend(sample_pool)
        print(f"Saved {len(processed_pool)} samples to {save_gen_path}")

        with open(save_gen_path, "w") as f:
            json.dump(processed_pool, f, indent=4)