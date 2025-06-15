INST = '''Here are the OpenAPI Specification of given APIs, including their http url, description and arguments.
{docs}

Based on provided APIs, please solve the question step by setp and write python code to call API and solve it. Try to write correct Python Code and avoid grammar error, e.g. `variable is not defined`.  You need to provide Python code that can be executed directly; Please add the name of the used APIs in Python comments for the attributable consideration. 
Note: you should be faithful to the question, please acquire any information you need by calling the APIs (e.g., person id or movie id). DO NOT make up value by yourself.

Here is an example to request the API:
```python
import requests
url = "http://0.0.0.0:8082/virtual"
data = "<The param dict>"
response = requests.post(url, data=json.dumps(data))
```

For each step, you need to state the problem you are trying to solve and provide the corresponding code.You can refer to the form below.

##Step 1: Write your Python code to make the first API call.
Python Code:
```python
[Please write the code.Each time you request a URL to obtain JSON data, you must print out the result of the request. There should be no other printing operations.]
```
[Step 1 Finished]

##Step 2: Process the data from the first API call if needed, and make any subsequent API calls if you need.
Python Code:
```python
[Write you code here.]
```
[Step 2 Finished]

##Step 3: Process the data from the second API call if needed, and make any subsequent API calls if you need.
```python
[Write you code here.]
```
[Step 3 Finished]

...

##Step X: Perform this step when you feel that you can already get the answer of user's query.
Parse the result from the API response, and print the final answer to the user's query.
Python Code:
```python
[Write you code here.]
```
[All Finished]
[Step X Finished]

**The number of steps to solve a problem is not fixed, and you can stop as soon as you feel that the user's problem can be solved.**
Note that I need to debug and improve the code with feedback from the compiler, so don't include any error handling mechanisms, such as try-catch statements.

Query: {query}
Your output:

Begin with 'Step 1:'
End with '[All Finished]\n[Step X Finished]', where 'X' is the last step number!!
**You only need to generate the code for the next step. Do not generate code for multiple steps.**
**Now generate you response strictly begin with '##Step 1', end with [Step 1 Finished]**
'''


CODETOOL_TMDB_INST = """{system}
Here are the OpenAPI Specification of given APIs, including their http url, description, arguments.
{docs}

If the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url.

Based on provided APIs, please solve the question step by setp and write python code to call API and solve it. Try to write correct Python Code and avoid grammar error, e.g. `variable is not defined`.  You need to provide Python code that can be executed directly; Please add the name of the used APIs in Python comments for the attributable consideration. 
Note: you should be faithful to the question, please acquire any information you need by calling the APIs (e.g., person id or movie id). DO NOT make up value by yourself.

You should use the following Http headers to call the API:
```python
headers = {headers}
```

Note: I will give you 'headers', do not make up one, just reference it in your code. Here is an example to request the API:
```python
import requests
url = "<The API url selected from the above APIs>"
params = "<The params dict>"
response = requests.get(url, headers=headers, params=params) # The variable `headers` has been defined
```

For each step, you need to state the problem you are trying to solve and provide the corresponding code.You can refer to the form below.

##Step 1: Write your Python code to make the first API call.
Python Code:
```python
[Please write the code.Each time you request a URL to obtain JSON data, you must print out the result of the request. There should be no other printing operations.]
```
[Step 1 Finished]

##Step 2: Process the data from the first API call if needed, and make any subsequent API calls if you need.
Python Code:
```python
[Write you code here.]
```
[Step 2 Finished]

##Step X: Perform this step when you feel that you can already get the answer of user's query.
Print the a complete final answer to the user's query.
Python Code:
```python
[Write you code here.]
```
[All Finished]
[Step X Finished]

**The number of steps to solve a problem is not fixed, and you can stop as soon as you feel that the user's problem can be solved.**
Note that I need to debug and improve the code with feedback from the compiler, so don't include any error handling mechanisms, such as try-catch statements.

Query: {query}
Your output:

Begin with 'Step 1:'
End with '[All Finished]\n[Step X Finished]', where 'X' is the last step number!!
**Ensure the code blocks are correctly started and ended with three backticks (```) for Step 1**
**You only need to generate the code for the next step. Do not generate code for multiple steps.**
**Now generate you response strictly begin with '##Step 1', end with [Step 1 Finished]**
"""


REWARD_MODEL_PROMPT_V2 = '''
# Task Description
The user is writing the code for API requests and parsing API responses to complete a query.
Your task is to evaluate the rationality of the code given by the user at each step.

# API Information
{system_info}
{his_info}
# You need to consider the following factors:
1. The API request made by the user at the current step is a necessary step to complete the query.
2. The names of the parameters for requesting the API should be strictly in line with the specified required_parameter.
3. The user needs to parse the content on the fields of the given response and should not create unknown field names.
4. The contribution of the user's API request at the current step to the completion of the query.

# The query is: 
"{user_query}"

# If you think the code written by the user meets the above requirements well, output "yes". Otherwise, output "no".
'''