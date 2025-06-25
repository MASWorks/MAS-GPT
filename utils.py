import json
import re
import ast
import os
import openai
import multiprocessing
import shutil
import random
import io                
import contextlib
import sys
import logging
import tempfile
import traceback
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

def handle_retry_error(retry_state):
    return None

class LLM():

    def __init__(self, model_list):
        self.model_list = model_list
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt, temperature=0.5):
        model_name, model_url, api_key = random.choice(self.model_list)
        llm = openai.OpenAI(base_url=f"{model_url}", api_key=api_key)
        try:
            completion = llm.chat.completions.create(
                model=f"{model_name}",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stop=['<|eot_id|>'],
                temperature=temperature,
                max_tokens=2048,
                timeout=600
            )
            raw_response = completion.choices[0].message.content
            # remove the think part for reasoning models such as deepseek-r1
            final_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            return final_response
        except Exception as e:
            logging.error(f"[Request Error] {e}")
            raise e


def parse_to_json(input_str):
    """
    Attempts to parse the input string into a JSON object.
    If direct parsing fails, extracts the first '{}' block and tries parsing it as JSON.
    
    Args:
        input_str (str): The input string to be parsed.
        
    Returns:
        dict: Parsed JSON object if successful.
        None: None if parsing fails.
    """
    try:
        # Attempt direct parsing
        return json.loads(input_str)
    except json.JSONDecodeError:
        # If direct parsing fails, search for the first '{}' block
        match = re.search(r'\{.*?\}', input_str, re.DOTALL)
        if match:
            json_fragment = match.group(0)
            try:
                # Attempt to parse the extracted block
                return json.loads(json_fragment)
            except json.JSONDecodeError:
                # Return none if the extracted block cannot be parsed
                return None
        else:
            # Return none if no '{}' block is found
            return None
        
def extract_code(text):
    """
    Extract code enclosed by triple backticks (```).

    Args:
        text (str): The input text containing code enclosed by triple backticks.

    Returns:
        str: Extracted code without language descriptors. An empty string if no matches found.
    """
    # Match content enclosed by triple backticks
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Extract the first match and strip surrounding whitespace
        match = matches[0].strip()
        # Split by lines
        lines = match.split("\n")
        # Check if the first line is a language descriptor (e.g., 'python', 'cpp', etc.)
        if len(lines) > 1 and lines[0].strip().lower() in {
            "python", "cpp", "java", "javascript", "c", "c++", "bash", "html", "css", "json", "sql"
        }:
            # Remove the first line if it is a language descriptor
            lines = lines[1:]
        code =  "\n".join(lines).strip()  # Join the remaining lines

        try:
            # Parse the code to check if it's valid Python syntax
            ast.parse(code)
            return code  # Code is valid and executable
        except (SyntaxError, ValueError):
            return ""  # Code is invalid or not executable
    
    return ""  # Return empty string if no matches found

def run_code_in_process(code, temp_dir, result_dict):
    """Execute code in a separate process."""
    try:
        # change current working directory to the temporary directory
        os.chdir(temp_dir)

        # global namespace
        global_vars = {
            'numpy': __import__('numpy'),
            'np': __import__('numpy'),
            'pandas': __import__('pandas'),
            'pd': __import__('pandas'),
            'matplotlib': __import__('matplotlib'),
            'plt': __import__('matplotlib.pyplot'),
            'math': __import__('math'),
            'random': __import__('random'),
            'os': __import__('os'),
            'sys': __import__('sys'),
            'abs': abs,
        }
        local_vars = {}

        # capture stdout
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, global_vars, local_vars)

        # put the result into the shared dictionary
        result_dict["stdout"] = stdout_capture.getvalue().strip()
        result_dict["output"] = local_vars.get("output", "None")
        result_dict["error"] = None

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # extract traceback information related to <string> (i.e. the code executed by exec)
        error_line = -1
        error_line_content = ""
        
        # iterate through the traceback stack
        for frame in traceback.extract_tb(exc_tb):
            if frame.filename == "<string>":
                error_line = frame.lineno
                # try to get the content of the line from the original code
                try:
                    error_line_content = code.splitlines()[error_line - 1]
                    error_line_content = f"The error occurred at line {error_line} in the code:\n{error_line_content.strip()}"
                except IndexError:
                    error_line_content = f"The error occurred at line {error_line}"
                break # stop when we find the frame we need

        # format the error report for LLM
        error_report = f"{exc_type.__name__}: {str(exc_value)}\n{error_line_content}"

        result_dict["error"] = error_report


def execute_code(code, timeout=30):
    """
    Execute Python code and capture standard output and `output` variable, execute code in the specified path, and clean up the directory after execution.

    Args:
        code (str): Python code to be executed.
        timeout (int): Maximum execution time (in seconds) for the code.

    Returns:
        str: String containing the print output and `output` variable value of the code execution.
    """

    if not code:
        return "Empty code. No output."
    
    # create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # use Manager().dict() to safely share results between processes
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # create and start a subprocess
    p = multiprocessing.Process(target=run_code_in_process, args=(code, temp_dir, result_dict))
    p.start()

    # wait for the process to end, or timeout
    p.join(timeout)

    final_result = ""

    # check if the process is still running (i.e. timeout)
    if p.is_alive():
        # force terminate the process!
        p.terminate()
        # wait for termination to complete
        p.join()
        final_result = "Execution Time Out"
    else:
        # process ended normally
        if result_dict.get("error"):
            final_result = f"Get the following error during code execution:\n{result_dict.get('error')}"
        else:
            final_result = f"Final output: {result_dict.get('output', 'None')}\nPrint during execution:\n{result_dict.get('stdout', '')}"

    # clean up the temporary directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[Warning] Error cleaning temp directory: {e}")

    return final_result

def _format_test_error_for_llm(exc_info, code_context, assert_str):
    """
    A helper function to format errors that occur during test execution for the LLM.
    It identifies the failing line within the user's function definition.
    """
    exc_type, exc_value, exc_tb = exc_info
    error_line = -1
    error_line_content = ""

    # Search the traceback for the frame executed within "<string>"
    for frame in traceback.extract_tb(exc_tb):
        if frame.filename == "<string>":
            error_line = frame.lineno
    
    if  error_line > 0:
        try:
            # The error is in the original code, not the assert statement
            error_line_content = code_context.splitlines()[error_line - 1]
            error_line_content = f"The error occurred at line {error_line} in the code:\n{error_line_content.strip()}"
        except IndexError:
            error_line_content = f"The error occurred at line {error_line}"

    # Format the error to be clear and concise for the LLM
    error_report = (
        f"The test case `{assert_str}` failed with an unexpected error:\n"
        f"  - Error: {exc_type.__name__}: {str(exc_value)}\n"
        f"  - {error_line_content}"
    )
    return error_report


def _execute_tests_in_process(code, test_cases, temp_dir_path, queue):
    """
    Worker function (runs in a separate process) to execute the code and test cases.
    """
    original_dir = os.getcwd()
    shared_context = {}
    
    try:
        os.chdir(temp_dir_path)

        # First, execute the user's code to define the function
        try:
            exec(code, shared_context)
        except Exception as e:
            # If the function definition itself has an error, report it and exit
            error_line = -1
            error_line_content = "Could not determine the failing line."
            exc_type_name = type(e).__name__
            exc_value_str = str(e)

            if isinstance(e, SyntaxError):
                # 对于 SyntaxError，直接从异常对象获取准确信息
                error_line = e.lineno
                # e.text 包含了出错的行内容，比从原代码中切分更可靠
                error_line_content = e.text if e.text else ""
                exc_value_str = e.msg # 使用 .msg 获取更简洁的错误信息
            else:
                # 对于其他运行时错误，使用 traceback 解析
                exc_tb = sys.exc_info()[2]
                # 寻找在 exec 代码中出错的最深层帧
                for frame in reversed(traceback.extract_tb(exc_tb)):
                    if frame.filename == "<string>":
                        error_line = frame.lineno
                        break
                if error_line > 0:
                    error_line_content = code.splitlines()[error_line - 1]
            
            error_report = (
                f"Your code failed to even define the function correctly.\n"
                f"  - Error: {exc_type_name}: {exc_value_str}\n"
                f"  - The error is at or near line {error_line} of your code:\n"
                f"    `{error_line_content.strip()}`"
            )
            queue.put((0, error_report))
            return

        # If definition is successful, run the test cases
        passed_count = 0
        feedback_reports = []
        for i, assert_str in enumerate(test_cases, 1):
            try:
                exec(assert_str, shared_context)
                passed_count += 1
                feedback_reports.append(f"✅ Test Case {i} Passed: `{assert_str}`")
            except AssertionError:
                feedback_reports.append(f"❌ Test Case {i} Failed: The assertion `{assert_str}` was not met.")
            except Exception:
                error_report = _format_test_error_for_llm(sys.exc_info(), code, assert_str)
                feedback_reports.append(f"💥 Test Case {i} Errored: {error_report}")
        
        # Combine all feedback into a final summary report
        if passed_count == len(test_cases):
             final_feedback = "✅ All test cases passed successfully."
        else:
            final_feedback = f"--- Test Execution Summary ---\n"
            final_feedback += f"Passed {passed_count} out of {len(test_cases)} test cases.\n\n"
            final_feedback += "\n".join(feedback_reports)

        queue.put((passed_count, final_feedback))

    finally:
        os.chdir(original_dir)


def test_code_get_feedback(code, test_cases, timeout=20):
    """
    Test the given code against a list of test cases in a specified directory with a time limit and provide feedback.

    Args:
        code (str): The Python code to be tested, typically a function definition.
        test_cases (list of str): A list of test cases, where each test case is an assert statement represented as a string.
        timeout (int): Maximum time (in seconds) allowed for testing all test cases.

    Returns:
        tuple: A tuple containing:
            - int: The number of test cases that passed.
            - str: A detailed, LLM-friendly feedback string.
    """
    if not code:
        return 0, "Empty code! This might be due to the code not being provided in the correct format (wrapped with triple backticks ```), causing extraction to fail."
    if not test_cases:
        return 0, "No test case provided!"

    # Create a unique temporary directory for this specific run
    temp_dir_path = tempfile.mkdtemp(prefix="test_workspace_")
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_execute_tests_in_process, args=(code, test_cases, temp_dir_path, queue))
    
    process.start()
    process.join(timeout)

    feedback = ""
    passed_count = 0

    if process.is_alive():
        process.terminate()
        process.join()
        passed_count = 0
        feedback = "Execution Time Out: The testing process took too long and was terminated."
    else:
        try:
            passed_count, feedback = queue.get_nowait()
        except multiprocessing.queues.Empty:
            passed_count = 0
            feedback = "Execution process finished unexpectedly without providing feedback."

    # Reliably clean up the temporary directory
    try:
        shutil.rmtree(temp_dir_path)
    except Exception as e:
        print(f"[Warning] Error cleaning temp directory: {e}")
    
    return passed_count, feedback


def websearch(query):
    """
    Search the internet given the query and return a list of passages.
    
    Args:
        query (str): a query or keyword for web search.
    Return:
        list: a list of searched passages(str)

    """
    return []

FUNCTION_SIGNATURE_DESIGNER_PROMPT = """Problem Description: {}

Task:
Given the problem description, write a Python function signature that matches the problem's requirements, \
including appropriate argument types. The function signature must include a brief and clear docstring that describes the function's purpose, its parameters, \
and the return value.

Your output must be formatted as a JSON object with two fields:
1. "think": Describe your reasoning and approach to solving the problem.
2. "function": Provide the function signature, including the docstring.

Use the following example as a guide for formatting:
{{
  "think": "Your reasoning process here.",
  "function": "def calculate_sum(a: int, b: int) -> int:\\n    \\\"\\\"\\\"\\n    Calculate the sum of two integers.\\n\\n    Parameters:\\n    a (int): The first integer.\\n    b (int): The second integer.\\n\\n    Returns:\\n    int: The sum of the two integers.\\n    \\\"\\\"\\\""
}}

Ensure the function signature and docstring are concise and directly aligned with the problem statement. You should output only the function signature so avoid including the function implementation. Avoid adding any text or explanations outside of the "think" field.

Please adhere strictly to the JSON format. Provide only the JSON object as the output.
"""

TEST_DESIGNER_PROMPT = """Problem Description: {problem}
Function Signature:
{function}

Task:
As a tester, your task is to create comprehensive test cases given the problem description and the function signature. \
These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability, in the format \
of assert statements. Remember to import necessary libs in each assert assert statements if necessary.

Your output must be formatted as a JSON object with four fields:
1. "think": Describe your reasoning and approach to solving the problem.
2. "basic": Several basic test cases to verify the fundamental functionality of the function under normal conditions.
3. "edge": Several edge test cases to evaluate the function's behavior under extreme or unusual conditions.
4. "large scale": Several large-scale test cases to assess the function's performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.

Use the following example as a guide for formatting:
{{
  "think": "Describe your reasoning and approach here.",
  "basic": [
    "# An ordinary case\\nassert sum([3,5]) == 8",
    "# An ordinary case\\nassert sum([2,7,3]) == 12",
    ...
  ],
  "edge": [
    "# Test with empty input list\\nassert sum([]) == 0",
    "# Test with single-element input\\nassert sum([7]) == 7",
    ...
  ],
  "large scale": [
    "# Test with large input list\\nlarge_list = [i for i in range(100)]\\nassert sum(large_list) == 4950",
    ...
  ]
}}

Please adhere strictly to the JSON format. Use '\\n' to represent line breaks in multi-line strings. Provide only the JSON object as the output. Do not add any text or explanations outside the JSON object. All comments must be included inside the JSON object as part of the strings. Do not place comments outside of the JSON structure to ensure proper parsing.
"""


def get_function_signature(llm, taskInfo):
    """
    Generate a Python function signature based on the problem description.

    Args:
        taskInfo (str): The problem description.

    Returns:
        str: The function signature with an appropriate docstring.
    """
    # Generates an instruction prompt by formatting the FUNCTION_SIGNATURE_DESIGNER_PROMPT with the task information
    function_signature_designer_instruction = FUNCTION_SIGNATURE_DESIGNER_PROMPT.format(taskInfo)
    # Calls the large language model (LLM) with the generated instruction
    answer = llm.call_llm(function_signature_designer_instruction)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Extracts and returns the function signature from the response
    if answer_dict and "function" in answer_dict.keys():
        return answer_dict["function"]
    return ""

# Function to generate test cases from the problem description and function signature
def get_test_cases(llm, taskInfo, function_signature):
    """
    Generate test cases based on the problem description and function signature.

    Args:
        taskInfo (str): The problem description.
        function_signature (str): The Python function signature.

    Returns:
        list: A list of test cases combining basic, edge, and large-scale scenarios.
    """
    # Generates an instruction prompt by formatting the TEST_DESIGNER_PROMPT with the task information and function signature
    test_designer_instruction = TEST_DESIGNER_PROMPT.format(problem=taskInfo, function=function_signature)
    # Calls the LLM with the generated instruction
    answer = llm.call_llm(test_designer_instruction, temperature=0.3)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Combines and returns the basic, edge, and large-scale test cases from the response
    if answer_dict and "basic" in answer_dict.keys() and "edge" in answer_dict.keys() and "large scale" in answer_dict.keys():
        return answer_dict["basic"] + answer_dict["edge"] + answer_dict["large scale"]
    # return an empty list if parse fails
    return []         

def extract_code_solution(solution):
    """
    Extract the code solution from the provided solution string.

    Args:
        solution (str): The solution string containing the code snippet.

    Returns:
        str: The extracted code snippet.
    """
    # Extract the code snippet enclosed by custom tags
    code_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>"
    match = re.search(code_pattern, solution, re.DOTALL)
    if match:
        code = match.group(1)
        # Remove code block tags if present
        code = re.sub(r"^```(?:\w+)?\n?|```$", "", code, flags=re.MULTILINE).strip()
        if code:
            return code
        return ""
    return ""

def generate_and_extract_code(llm, prompt, temperature=None, max_attempts=3):
        """
        Generate a response from the LLM and extract the contained code with retry logic.

        This function attempts to generate a response from the LLM containing a code snippet.
        It first extracts the portion of the response wrapped within custom tags (e.g., <Code Solution>). 
        Then remove possible code block tags (e.g., ```python).
        Returns both the full response and the extracted code. 
        If no valid code is found after multiple attempts, it returns the last response and an empty string for the code.

        Args:
            prompt (str): The instruction to send to the LLM to generate a response with code.
            temperature (float, optional): Sampling temperature for the LLM, controlling randomness in the output.
            max_attempts (int): Maximum number of attempts to fetch a response with valid code. Default is 3.
            
        Returns:
            tuple:
                str: The full LLM response.
                str: The extracted code snippet, or an empty string if no valid code is detected.
        """
        attempts = 0  # Track the number of attempts
        tag_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>" # Regular expression pattern to extract content within custom tags
        
        while attempts < max_attempts:
            # Generate response using the LLM
            if temperature:
                llm_response = llm.call_llm(prompt, temperature=temperature)
            else:
                llm_response = llm.call_llm(prompt)
                
            code = extract_code_solution(llm_response)
            if code:
                return llm_response, code
            
            attempts += 1  # Increment attempts and retry if no valid code is detected
        
        # Return the last LLM response and an empty code snippet after exhausting all attempts
        return llm_response, ""

if __name__ == '__main__':
    # normal execution
    good_code = "output = sum([i for i in range(10)])\nprint('Calculation done.')"
    print("--- Running good code ---")
    print(execute_code(good_code))
    print("\n" + "="*30 + "\n")

    # timeout execution
    timeout_code = "import time\nprint('Starting infinite loop...')\nwhile True:\n    time.sleep(1)"
    print("--- Running timeout code (will wait for 5 seconds) ---")
    print(execute_code(timeout_code, timeout=5))
    print("\n" + "="*30 + "\n")

    # error execution
    error_code = "a = 1\nx = 1 / 0"
    print("--- Running error code ---")
    print(execute_code(error_code))

    # error execution
    error_code = "my_list = [10, 20, 30]\noutput = my_list[3]"
    print("--- Running error code ---")
    print(execute_code(error_code))

    # test code get feedback
    def run_test_and_print(title, code, test_cases, timeout=20):
        print(f"--- {title} ---")
        passed_count, feedback = test_code_get_feedback(code, test_cases, timeout)
        print(f"Result: {passed_count} test(s) passed.")
        print("Feedback:")
        print(feedback)
        print("-" * (len(title) + 4) + "\n")

    # Test Case 1: All Pass
    code_all_pass = """
def add(a, b):
    # A simple function to add two numbers
    return a + b
"""
    tests_all_pass = [
        "assert add(2, 3) == 5",
        "assert add(-1, 1) == 0",
        "assert add(0, 0) == 0"
    ]
    run_test_and_print("Test Case 1: All Pass", code_all_pass, tests_all_pass)

    # Test Case 2: Assertion Failure
    tests_assertion_fail = [
        "assert add(2, 3) == 5",
        "assert add(5, 5) == 99 # This assertion is wrong"
    ]
    run_test_and_print("Test Case 2: Assertion Failure", code_all_pass, tests_assertion_fail)

    # Test Case 3: Unexpected Runtime Error
    tests_runtime_error = [
        "assert add(2, 3) == 5",
        "assert add(5, 'a') == 5 # This will cause a TypeError"
    ]
    run_test_and_print("Test Case 3: Unexpected Runtime Error", code_all_pass, tests_runtime_error)

    # Test Case 4: Syntax Error in Definition
    code_syntax_error = """
def add(a, b) # Missing colon at the end
    return a + b
"""
    tests_for_bad_code = ["assert add(1, 1) == 2"]
    run_test_and_print("Test Case 4: Syntax Error in Definition", code_syntax_error, tests_for_bad_code)
    
    # Test Case 5: Execution Timeout
    code_timeout = """
import time
def slow_function():
    time.sleep(1)
    while True:
        # Infinite loop to cause a timeout
        pass
    return "done"
"""
    tests_for_timeout = ["assert slow_function() == 'done'"]
    # We use a short timeout to demonstrate the feature
    run_test_and_print("Test Case 5: Execution Timeout", code_timeout, tests_for_timeout, timeout=3)