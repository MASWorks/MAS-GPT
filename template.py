def form_instruction(query):
    instruction = "I will give a user query and a code template for multi-agent system. You need to generate a proper multi-agent system described in Python code to solve the query. The Python code should be able to take the query as input and return the answer as output. The code should be able to run without any errors. Please make sure the code is well-structured and easy to understand.\n\n"
    instruction += f"Here is the user query: {query}\n\n"
    instruction += "Here is the code template:\n\n"
    instruction += '''```python
from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        pass
        return
```'''
    return instruction

SYSTEM_PROMPT = """You are an expert in generating multi-agent systems. You need to generate a proper multi-agent system described in Python code to solve the user query. The Python code should be able to take the user query as input and return the answer as output. The code should be able to run without any errors. Please make sure the code is well-structured and easy to understand.

## Here is an code template for multi-agent system:
```python
from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        pass
        return
```

## The utils library contains the following elements that you can use in your code:
- `LLM(model_list)`: a class that represents a large language model with the given model list.
- `execute_code(code)`: a function that executes the given code and returns the output.
- `test_code_get_feedback(code, test_cases)`: a function that tests the given code with the test cases and returns the feedback.
- `get_function_signature(llm, taskInfo)`: a function that returns the generated function signature for the given task.
- `get_test_cases(llm, taskInfo, function_signature)`: a function that returns the generated test cases for the given task and function signature.
- `extract_code_solution(solution)`: a function that returns the code by extracting (wrapped within <Code Solution> and </Code Solution>) from the given solution.
- `generate_and_extract_code(llm, prompt, temperature=None)`: a function that returns the generated response and the extracted code from the response.

If you need to use other functions, you need to implement them by yourself in your returned code."""