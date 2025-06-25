import os
import json
import sys
import concurrent.futures
import logging
from tqdm import tqdm
import traceback
import types
import threading
import multiprocessing
import argparse

def run_mas(sample, model_list):
    query, mas_str = sample["query"], sample["mas_str"]
    module = types.ModuleType("mas_module")
    exec(mas_str, module.__dict__)

    if 'MAS' in module.__dict__:
        MAS = module.MAS
        mas_instance = MAS(model_list=model_list)
        if hasattr(mas_instance, "forward") and callable(getattr(mas_instance, "forward")):
            response = mas_instance.forward(query)
            if response is None:
                logging.error(f"Query: {query}\n|| {mas_str} failed to execute: response is None")
                return None
            else:
                return response
        else:
            logging.error("MAS class does not have a callable 'forward' method.")
    else:
        logging.error("Module does not have a 'MAS' class.")
    return None

def run_mas_wrapper(queue, sample, model_list):
    """
    A wrapper function for running run_mas in a separate process and putting the result into a queue.
    """
    try:
        response = run_mas(sample, model_list)
        queue.put(("success", response))
    except Exception as e:
        # Capture any exceptions inside run_mas and return them
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        queue.put(("error", error_info))

def get_backup_mas():
    return "from utils import *\n\nclass MAS():\n    def __init__(self, model_list):\n        self.llm = LLM(model_list)\n\n    def forward(self, taskInfo):\n        \"\"\"\n        Design a multi agent system for reasoning and task solving.\n        5 chain-of-thought agents are created for diverse solutions, and a final decision-making agent is used to generate the final answer.\n        \"\"\"\n\n        # Step-by-step instruction for each chain-of-thought agent to reason and generate answer\n        cot_instruction = f\"Task: {taskInfo}\\n\\nPlease think step by step and then solve the task.\"\n\n        # Set the number of solutions to generate; using 5 for variety and diversity\n        N = 5\n        # Call the llm to generate each solution\n        cot_results = [self.llm.call_llm(cot_instruction) for _ in range(N)]  \n\n        # Get the instruction for the final decision-making agent based on all generated solutions\n        final_decision_instruction = self.get_final_decision_instruction(taskInfo, cot_results)\n\n        # Call the llm to process the final decision-making instruction and generate the final answer\n        final_decision_result = self.llm.call_llm(final_decision_instruction)\n\n        # Return the final solution\n        return final_decision_result\n\n    def get_final_decision_instruction(self, taskInfo, cot_results):\n        \"\"\"\n        Format an instruction for final decision-making based on a given task description and a list of solutions.\n\n        Parameters:\n        taskInfo (str): A description of the task that needs to be completed.\n        cot_results (list): A list containing solutions or reasoning steps for the task.\n\n        Returns:\n        str: A formatted instruction that includes the task description, each solution, and a prompt for final decision-making.\n        \"\"\"\n\n        # Initialize the instruction text with a general guideline\n        instruction = f\"Task:\\n{taskInfo}\\n\\n\"\n\n        # Append each solution from cot_results to the instruction\n        for i, result in enumerate(cot_results):\n            instruction += f\"Solution {i+1}:\\n{result}\\n\\n\"  # Number each solution for clarity\n\n        # Add the final prompt to encourage reasoning over the solutions and provide a final answer\n        instruction += \"Given all the above solutions, reason over them carefully and provide a final answer to the task.\"\n        \n        # Return the complete instruction text\n        return instruction"

def process_sample(sample, model_list, output_json, file_lock, timeout=600, max_retries=1):

    for attempt in range(1 + max_retries):
        is_retry = attempt > 0
        
        if is_retry:
            logging.warning(f"Timeout occurred for query: {sample['query']}. Retrying with fallback MAS script (Attempt {attempt})...")
            print(f"Timeout occurred for query: {sample['query']}. Retrying with fallback MAS script (Attempt {attempt})...")
            try:
                sample["mas_str"] = get_backup_mas()
            except Exception as e:
                logging.error(f"Failed to read fallback script for query: {sample['query']}. Error: {e}. Skipping sample.")
                break
        
        # Use Queue to safely pass data between processes
        q = multiprocessing.Queue()
        # Create a standard, non-daemon process
        p = multiprocessing.Process(target=run_mas_wrapper, args=(q, sample, model_list))

        try:
            p.start()
            # Wait for the process to complete or timeout
            p.join(timeout)

            # Check if the process is still alive (i.e. timeout)
            if p.is_alive():
                p.terminate() # Force termination
                p.join() # Wait for termination to complete
                # Raise our own timeout error, enter retry logic
                raise multiprocessing.TimeoutError 

            # Get the result from the queue
            status, result = q.get()
            
            if status == "success":
                response = result
                if response is not None:
                    with file_lock:
                        with open(output_json, "a") as result_file:
                            sample["generated_output"] = response
                            json.dump(sample, result_file)
                            result_file.write("\n")
                else:
                    logging.error(f"Response is None for query: {sample['query']}. Skipping.")
                return

            else: # status == "error"
                # An error occurred inside run_mas
                logging.error(f"An error occurred inside run_mas for query: {sample['query']}.")
                logging.error(f"Traceback from subprocess:\n{result['traceback']}")
                break # An error occurred, terminate retry

        except multiprocessing.TimeoutError:
            if attempt >= max_retries:
                logging.error(f"Final attempt timed out for query: {sample['query']}. Skipping.")
                print(f"Final attempt timed out for query: {sample['query']}. Skipping.")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred in process_sample for query: {sample['query']}.")
            logging.error(f"Traceback: {traceback.format_exc()}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # The name of the MAS-GPT model, same as the name in masgpt_mas_gen.py
    parser.add_argument("--masgpt_name", type=str, default="MAS-GPT-32B")

    # The model used to drive the agent behind multi-agent system
    parser.add_argument("--mas_model_name", type=str, default="llama-3-70b-instruct", help="The agent backend to be used for inference.")
    parser.add_argument("--mas_model_config", type=str, default="./model_configs/model_config.json")

    parser.add_argument("--test_dataset_names", type=str, nargs='+', default=["MATH", "GSM-Hard"])
    
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    print("="*50)
    print(json.dumps(vars(args), indent=4))
    for i, test_dataset_name in enumerate(args.test_dataset_names):

        print(f">> Processing {i}-th dataset: {test_dataset_name}")

        # ================== Define the output files ==================
        masgpt_gen_path = f"./test/{test_dataset_name}/masgpt_{args.masgpt_name}_gen.json"
        output_logging = f"test/{test_dataset_name}/{args.mas_model_name}/log/masgpt_{args.masgpt_name}_infer.txt"
        output_json = f"test/{test_dataset_name}/{args.mas_model_name}/masgpt_{args.masgpt_name}_infer.jsonl"
        os.makedirs(os.path.dirname(output_logging), exist_ok=True)

        # ================== Load the test query pool ==================
        with open(masgpt_gen_path, "r") as f:
            sample_pool = json.load(f)
        sample_pool = sample_pool[:5] if args.dry_run else sample_pool
        print(f">> Initially: {len(sample_pool)} samples")

        # filter out processed queries
        processed_queries = set()
        if os.path.exists(output_json):
            with open(output_json, "r") as f:
                for line in f:
                    infered_sample = json.loads(line)
                    processed_queries.add(infered_sample["query"])

        sample_pool = [sample for sample in sample_pool if sample["query"] not in processed_queries]
        print(f">> After filtering: {len(sample_pool)} samples")

        # ================== Define the model list ==================
        with open(args.mas_model_config, "r") as f:
            config = json.load(f)
            model_dict = config["model_dict"]
            worker_dict = config["worker_dict"]

        model_list = model_dict[args.mas_model_name]
        max_workers = worker_dict[args.mas_model_name] * len(model_list)

        logging.basicConfig(filename=output_logging, level=logging.INFO, format='%(asctime)s - %(message)s')

        file_lock = threading.Lock()

        if args.sequential:
            for sample in tqdm(sample_pool, desc="Processing queries"):
                process_sample(sample, model_list, output_json, file_lock, timeout=args.timeout)
        else:
            # Use ThreadPoolExecutor to process samples in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                from functools import partial
                process_func = partial(process_sample, model_list=model_list, output_json=output_json, file_lock=file_lock, timeout=args.timeout)
                for _ in tqdm(executor.map(process_func, sample_pool), total=len(sample_pool), desc="Processing queries"):
                    pass