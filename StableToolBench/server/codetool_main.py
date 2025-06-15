from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.requests import Request
import uvicorn
import time
from datetime import datetime
import json
import os, yaml, ast, re
import requests
from typing import Union
from utils import standardize, change_name

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from tenacity import retry, wait_random_exponential, stop_after_attempt

config_file='config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
print(CONFIG)
CACHE_FOLDER = CONFIG['cache_folder']
LOG_FILE = CONFIG['log_file']
# # OpenAI API
from openai import OpenAI

def get_openai_res(messages):
    client = OpenAI(
        api_key=CONFIG['api_key'],
        base_url=CONFIG['api_base'],
    )
    response = client.chat.completions.create(
        model = CONFIG['model'],
        temperature = CONFIG['temperature'],
        messages = messages,
        timeout = 600,
        )
    
    response = [response.choices[0].message.content]

    return response

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str
    toolbench_key: str

def prepare_tool_name_and_url(info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name)).split(f"_for_{tool_name}")[0]
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from my_tools.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from my_tools.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string

def write_log(request, response, type):
    log = """\
>>>>>>>>>>>>>>>>>>>>>>>
TIME: {curr_time}
TYPE: {type}
REQUEST: {request}
RESPONSE: {response}
<<<<<<<<<<<<<<<<<<<<<<<
"""
    with open(LOG_FILE, "a") as f:
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(log.format(curr_time=curr_time, type=type, request=request, response=response))


@app.post('/virtual')
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))
def get_virtual_response(request: Request, info: Info):
    user_key = info.toolbench_key

    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(info)
    tool_input = info.tool_input
    tool_name_original = info.tool_name

    if api_name == "chat_with_user":
        write_log(request=info, response=real_response, type="chat_with_user")
        return {"error": "", "response": "Chat with user."}
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        elif isinstance(tool_input, dict):
            tool_input = tool_input
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            print(type(tool_input))
            print(tool_input)
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            write_log(request=info, response=response_dict, type="tool_input_parse_error")
            return response_dict
    if not os.path.exists(CACHE_FOLDER):
        os.mkdir(CACHE_FOLDER)

    # load from cache
    cache = {}
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'toolbench_key': user_key
    }
    os.environ['HTTP_PROXY']= ''
    if "_for_" in tool_name_original:
        tool_name_real = tool_name_original.split("_for_")[0]
    else:
        tool_name_real = tool_name_original
    data = {
        "category": standard_category,
        "tool_name": tool_name_real,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": "",
        "toolbench_key": user_key
    }
    # prerequisite: to read files correctly, "my_tools_cache" folder and "toolenv/tools/" folder should be available
    # The cache loaded from the local storage needs to undergo format conversion so that it can be parsed in the form of programmatic tool invocation.
    try:
        if os.path.exists(os.path.join(CACHE_FOLDER, standard_category)):
            if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name)):
                if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json")):
                    tools_cache_record = json.load(open(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json"), "r"))
                    cache.update(tools_cache_record)
                    if str(tool_input) in cache:
                        print("using cached real response")
                        response_dict = cache[str(tool_input)]
                        if 'response' in response_dict:
                            response_value = response_dict['response']
                            if isinstance(response_value, (list, dict)):
                                print('valid cache response')
                            if isinstance(response_value, str):
                                if is_valid_json(response_value):
                                    try:
                                        response_list = ast.literal_eval(response_value)
                                        response_dict['response'] = response_list
                                        # return response_list
                                    except (ValueError, SyntaxError):
                                        response_list = json.loads(response_value)
                                        response_dict['response'] = response_list
                                else:
                                    response_dict = reformat_response(response_dict)

                        save_cache(cache, tool_input, response_dict, standard_category, tool_name, api_name) 
                        write_log(request=info, response=response_dict, type="cached_real_response")
                        return response_dict
    except Exception as e:
        print(f"Loading cache error: {e}")     

    """
    Call the real api before generating fake response
    """
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'toolbench_key': user_key
    }
    os.environ['HTTP_PROXY']= ''
    if "_for_" in tool_name_original:
        tool_name_real = tool_name_original.split("_for_")[0]
    else:
        tool_name_real = tool_name_original
    data = {
        "category": standard_category,
        "tool_name": tool_name_real,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": "",
        "toolbench_key": user_key
    }

    real_response = requests.post(CONFIG['toolbench_url'], headers=headers, data=json.dumps(data))
    print(real_response)
    # Check if the request was successful
    if real_response.status_code == 200:
        real_response = real_response.json()
        if 'response' in real_response:
            response_value = real_response['response']
            if isinstance(response_value, str):
                if is_valid_json(response_value):
                    print('valid cache real response')
                    try:
                        response_list = ast.literal_eval(response_value)
                        real_response['response'] = response_list
                        # return response_list
                    except (ValueError, SyntaxError):
                        response_list = json.loads(response_value)
                        real_response['response'] = response_list
                    return real_response
                else:
                    real_response = reformat_response(real_response)
                    save_cache(cache, tool_input, real_response, standard_category, tool_name, api_name)
                    return real_response

        # if check_result(real_response):
        #     print("returning real_response")
        #     write_log(request=info, response=real_response, type="real_response")
        #     if CONFIG['is_save']:
        #         print('进行save')
        #         save_cache(cache, tool_input, real_response, standard_category, tool_name, api_name)
        #     return real_response
        
    """
    Fake response function here. Use the cached history response for in-context examples.
    result = fake_response_function(api_doc, api_name, api_parameters, *kwargs)
    """

    # parse api_doc
    tool_name_original = standardize(tool_name_original)
    api_name = standardize(api_name)
    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    try:
        if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category)):
            if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json")):
                # read json
                api_intro = json.load(open(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json"), "r"))
                # get tool_dexcription and api_info
                tool_description = api_intro['tool_description']
                api_info = []
                for api in api_intro['api_list']:
                    if api_name == standardize(api['name']):
                        api_info.append({
                            'name': api['name'],
                            'description': api['description']
                        })
                # check invalid api name
                if len(api_info) == 0:
                    print("cant match api name")
                api_doc = {
                    'tool_description': tool_description,
                    'api_info': api_info
                }
            else:
                print(f"cant get {tool_name_original}")
    except Exception as e:
        print(f"Loading api_doc error: {e}")

    # get several examples from cache
    example_num = 5
    # get top example_num examples
    api_example = list(cache.items())[:example_num]
    while len(str(api_example)) > 2048 and example_num > 1:
        example_num -= 1
        api_example = list(cache.items())[:example_num]

    print(f"api example: {api_example},,, tool_input: {tool_input},,, api_doc: {api_doc},")
        
    result = fake_response_function_chat(api_example,tool_input,api_doc)
    print(f"fake result: {result}")

    if CONFIG['is_save']:
        save_cache(cache, tool_input, result, standard_category, tool_name, api_name)
    write_log(request=info, response=result, type="fake_response")
    if not isinstance(result, dict):
        return json.loads(result)
    else:
        return result


def reformat_response(real_response):
    RE_FORMAT = '''The following is an API response in JSON format:
{api_response} 
However, the list or dict in its "response" field is not in a correct format. 
You need to delete certain content to make it into a correct list or dict format. After deleting the certain content, it is necessary to ensure that the formats of the elements in the list or dict are consistent.
Don't add any other content!
Your responses must adhere to a specific JSON structure, which is as follows:
```json
{{"error": "", "response": a correct list or dict format}}
```
You only need to output the final content in JSON format.
'''
    # print(real_response)
    if real_response['response'].startswith(("{", "[")):
        messages = [
            {"role": "user", "content": RE_FORMAT.format(api_response=json.dumps(real_response))}
        ]
        reply = get_openai_res(messages)

        max_retries = 2
        for attempt in range(max_retries):
            if '```json' in reply[0]:
                pattern = r'```json([\s\S]*?)```'
                match = re.search(pattern, reply[0])
                if match:
                    json_content = match.group(1).strip()
                    real_response = json.loads(json_content)
                    print('reformat cache real response')
                break
            else:
                reply = get_openai_res(messages)

    
    return real_response


def is_valid_json(result):
    """
    Checks if the given string is valid JSON.

    Args:
      data: The string to be checked.

    Returns:
      True if the string is valid JSON, False otherwise.
    """
    # check json format
    try:
        result = json.loads(result)
        return True
    except Exception as e:
        # print(f"Can not parse result into json: {result}")
        return False
    
def recursive_parse(data):
    try:
        parsed_data = ast.literal_eval(data)
        return parsed_data
    except (ValueError, SyntaxError):
        pass
    try:
        parsed_data = json.loads(data)
        if isinstance(parsed_data, dict):
            return {key: recursive_parse(value) for key, value in parsed_data.items()}
        elif isinstance(parsed_data, list):
            return [recursive_parse(element) for element in parsed_data]
        else:
            return parsed_data
    except json.JSONDecodeError:
        return data


def check_result(processes_value: dict):
    if 'error' not in processes_value or processes_value['error'] != '':
        return False
    if 'response' not in processes_value:
        return False
    response = str(processes_value['response'])
    if 'got an unexpected keyword argument' in response.lower():
        return True
    # elif 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or '404' in response.lower() or '504' in response.lower() or '500' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
    elif 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
        return False
    elif 'authoriz' in response.lower() or 'authenticat' in response.lower() or 'unauthorized' in response.lower() or 'blocked user' in response.lower() or 'unsubscribe' in response.lower() or 'blocked' in response.lower() or '401' in response.lower() or '403' in response.lower() or 'credential' in response.lower() or 'unauthenticated' in response.lower() or 'disabled for your subscription' in response.lower() or 'ACCESS_DENIED' in response or 'invalid consumer key' in response.lower():
        return False
    elif 'parameter' in response.lower() or 'parse' in response.lower() or 'is not defined' in response.lower():
        return False
    elif len(response) == 0:
        return False
    elif "status_code=50" in response or "status_code=429" in response:
        return False
    return True

def save_cache(cache, tool_input, result, standard_category, tool_name, api_name, save_folder=CACHE_FOLDER):
    # save cache
    try:
        if isinstance(result, dict):
            cache[str(tool_input)] = result
        elif isinstance(result, str):
            try:
                result_dict = json.loads(result)
                cache[str(tool_input)] = result_dict
            except Exception as e:
                print(f"Load result failed: {e}")
                return

        if not os.path.exists(os.path.join(save_folder, standard_category)):
            os.mkdir(os.path.join(save_folder, standard_category))
        if not os.path.exists(os.path.join(save_folder, standard_category, tool_name)):
            os.mkdir(os.path.join(save_folder, standard_category, tool_name))    
        json.dump(cache, open(os.path.join(save_folder, standard_category, tool_name, api_name+".json"), "w"), indent=4)
    except Exception as e:
        print(f"Save cache failed: {e}")

def fake_response_function_chat(api_example, tool_input, api_doc):
    '''
    api_example: list of tuple, [(input, output), ...]
    tool_input: dict, input of the tool
    api_doc: dict, api document
    '''
    system_prompt = '''
Imagine you are an API Server operating within a specialized tool, which contains a collection of distinct APIs. Your role is to deeply understand the function of each API based on their descriptions in the API documentation. As you receive specific inputs for individual API calls within this tool, analyze these inputs to determine their intended purpose. Your task is to craft a JSON formatted response that aligns with the expected output of the API, guided by the provided examples.\n
Your responses must adhere to a specific JSON structure, which is as follows:\n
{
    "error": "",
    "response": "<Your_Response>"
}\n
The error field should remain empty, indicating no errors in processing. The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality. If the provided examples are mostly error messages or lack substantial content, use your judgment to create relevant and accurate responses. The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output within the tool.\n
Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
Note that:
- your response should be around 100 to 200 words, containing rich information given the api input parameters. Keep Your answer short and simple.
- your response must be effective and have practical content.
- if the api response example if null or ineffective, ignore the example and give your independent response.

API calls may fail for various reasons, such as invalid input parameters, authentication issues, or server errors. Your goal is to generate a response that accurately reflects the API's intended functionality, even if the input parameters are incorrect. Your response should be informative and relevant to the API's purpose, providing a clear and concise explanation of the expected output based on the input provided.
Here is an example:
API doc
{
    "name": "properties/get-broadband",
    "url": "https://zoopla.p.rapidapi.com/properties/get-broadband",
    "description": "Get broadband information",
    "method": "GET",
    "required_parameters": [
        {
            "name": "listing_id",
            "type": "NUMBER",
            "description": "The value of listing_id field returned in .../properties/list endpoint",
            "default": "56354192"
        }
    ],
    "optional_parameters": [],
    "code": "import requests\n\nurl = \"https://zoopla.p.rapidapi.com/properties/get-broadband\"\nquerystring = {\"listing_id\": \"56354192\"}\n\nheaders = {\n            \"X-RapidAPI-Key\": \"SIGN-UP-FOR-KEY\",\n            \"X-RapidAPI-Host\": \"zoopla.p.rapidapi.com\"\n        }\n\nresponse = requests.get(url, headers=headers, params=querystring)\nprint(response.json())\n",
    "convert_code": "import requests\n\nurl = \"https://zoopla.p.rapidapi.com/properties/get-broadband\"\nquerystring = {\"listing_id\": \"56354192\"}\n\nheaders = {\n            \"X-RapidAPI-Key\": \"SIGN-UP-FOR-KEY\",\n            \"X-RapidAPI-Host\": \"zoopla.p.rapidapi.com\"\n        }\n\nresponse = requests.get(url, headers=headers, params=querystring)\nprint(response.json())\n",
    "test_endpoint": "",
    "statuscode": 200,
    "schema": {}
}
Request:
    data = {
        "category": "Business",
        "tool_name": "zoopla_v2",
        "api_name": "properties_get_broadband",
        "tool_input": {'listing_id': '456789', "abdc": 11123},
        "strip": "",
        "toolbench_key": "xxx"
    }
Response:
    {"error":"Function executing from toolenv.tools.Business.zoopla_v2.api import properties_get_broadband error...\nproperties_get_broadband() got an unexpected keyword argument 'abdc'","response":""}


Your will also be given successful examples of API calls and their expected outputs, based on which you will generate the response for the given input.
    '''
    system_prompt = {"role": "system", "content": system_prompt}
    # user prompt, truncated to 2048 characters if too long
    user_prompt = "API Documentation:"+str(api_doc)+"\n"+"API Examples:"+str(api_example)[:2048]+"\n"+"API Input:"+str(tool_input)+"\n"
    user_prompt = {"role": "user", "content": user_prompt}

    max_retries = 3 
    flag = False
    for attempt in range(max_retries):
        response = get_openai_res(messages=[system_prompt, user_prompt])
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        if is_valid_json(response):
            flag = True
            break
        print(f"Invalid JSON response on attempt {attempt + 1}. Retrying...")
        time.sleep(1)  # Optional delay between retries

    if flag:
        return response
    else:
        fake_error = {
            "error": "Failed to generate fake response",
            "response": "",
        }
        return json.dumps(fake_error)

if __name__ == "__main__":
    uvicorn.run(app="codetool_main:app", host="0.0.0.0", port=CONFIG['port'])