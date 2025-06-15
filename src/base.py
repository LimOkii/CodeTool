import json
import yaml
from tqdm import tqdm
from src.instruction import *

config = yaml.safe_load(open('src/config.yaml', 'r'))

def simplify_response_template(data):
    if 'required' in data and 'properties' in data:
        for k, v in data['properties'].items():
            if k not in data['required']:
                data.pop(k)
    if 'type' in data and data['type'] == 'object' and 'properties' in data:
        for k, v in data['properties'].items():
            data['properties'][k] = simplify_response_template(v)
    else:
        for k, v in data.items():
            if k in ['example', 'nullable', 'x-spotify-docs-type']:
                data.pop(k)
            if k == 'description':
                data[k] = normalize(v)
    return data


def simplify_spec(data):
    """
    Recursively simplify the dictionary by removing specific keys.

    :param data: The input dictionary to be simplified.
    :return: A simplified dictionary with specified keys removed.
    """
    keys_to_remove = ['example', 'nullable', 'x-spotify-docs-type', 'required', 'default', 'minimum', 'maximum', 'examples']

    if isinstance(data, dict):
        results = {}
        for k, v in data.items():
            if k in keys_to_remove:
                continue
            # if k == 'description':
            #     results[k] = normalize(simplify_spec(v))
            # else:
            results[k] = simplify_spec(v)
        return results
    elif isinstance(data, list):
        return [simplify_spec(item) for item in data]
    else:
        if type(data) == str:
            return normalize(data)
        return data


def normalize(sss):
    for s in ['<br />', '<br/>', '_**NOTE**:']:
        sss = sss.replace(s, '\n')
    sss = sss.split('\n')[0]
    tmp = [
        '(/documentation/web-api/#spotify-uris-and-ids)',
        '(https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)',
        '(https://www.spotify.com/se/account/overview/)',
        '(http://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)',
        '<br/>',
        '<br>',
        '<br />',
        '\n',
        '/documentation/general/guides/track-relinking-guide/',
        '(http://en.wikipedia.org/wiki/Universal_Product_Code)',
        '(http://en.wikipedia.org/wiki/International_Standard_Recording_Code)',
        '/documentation/web-api/#spotify-uris-and-ids'
    ]
    for s in tmp:
        sss = sss.replace(s, '')

    for i in range(10):
        sss = sss.replace(f'[{i}].', '')
        sss = sss.replace(f'[{i}]', '')
    return sss.strip()


class Base:

    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name
        self.token = []

    def normalize(self, sss):
        return sss

    # def generate(self, messages):
    #     res = get_from_openai(model_name=self.model_name, messages=messages, usage=True)
    #     self.token.append(res['usage'])
    #     return self.normalize(res['content']), res['usage']

    def get_token(self):
        tmp = []
        for line in self.token:
            tmp = [e1 + e2 for e1, e2 in zip(tmp, line)]
        return tmp
    

class Tool:
    def __init__(self, spec: dict):

        self.name = spec['name']
        self.method = spec['method']
        self.url = spec['url']
        self.description = spec['description']
        self.parameter = spec['parameters'] if 'parameters' in spec else []
        self.responses = {}

        if 'requestBody' in spec and spec['requestBody'] != None:
            self.requestBody = simplify_spec(spec['requestBody']['content']['application/json']["schema"]['properties'])
        else:
            self.requestBody = 'This API do not need the request body when calling.'

        if 'responses' in spec and spec['responses'] is not None and 'content' in spec['responses']:
            self.responses['responses'] = simplify_spec(spec['responses']['content']['application/json']["schema"]['properties'])
            self.responses['responses'] = json.dumps(self.responses['responses'], indent=4)
        else:
            self.responses['responses'] = 'This API has no return value.'

        if '_responses_json' in spec and spec['_responses_json'] is not None:
            self.responses['_responses_json'] = json.dumps(spec['_responses_json'], indent=4) if type(spec['_responses_json']) == dict else spec['_responses_json']
        else:
            self.responses['_responses_json'] = None

    def update_response(self, response_format, response_example):
        if response_format == '_response_yaml':
            self.responses[response_format] = response_example
        else:
            self.responses[response_format] = response_example if type(response_example) == str else json.dumps(response_example, indent=4)

    def get_parameters(self) -> str:
        if len(self.parameter) == 0:
            parameter = 'No extra parameter, just replace the `{variable}` in the url path with actual value.'
        else:
            parameter = []
            for p in self.parameter:
                tmp = "- " + p['name'] + ": " + normalize(p['description'])
                if 'schema' in p and 'type' in p['schema']:
                    tmp += " (type: " + p['schema']['type'] + ")"
                parameter.append(tmp)
            parameter = '\n'.join(parameter)
            if '{' in self.url:
                parameter += '\nThe `{variable}` in the url path should also be replaced with actual value.'
        return parameter

    def formulate(self, is_description=True, is_parameters=True, is_request_type=True, is_url=True,
                  execution_results_type=None, is_request_body=True):
        text_doc = ["""API name: """ + self.name]
        if is_url:
            text_doc.append('### API url\n' + self.url)
        if is_request_type:
            method = """### Request type\n""" + self.method
            text_doc.append(method)
        if is_description:
            # description = """### Description\n""" + normalize(self.description)
            description = """### Description\n""" + self.description
            text_doc.append(description)
        if is_parameters:
            parameters = '### Parameter\n' + self.get_parameters()
            text_doc.append(parameters)
        # if execution_results_type is not None and execution_results_type in self.responses:
        #     response = '### Execution result specification\n' + str(self.responses[execution_results_type])
        #     text_doc.append(response)
        if is_request_body:
            requestBody = '### Request body\n' + json.dumps(self.requestBody, indent=4)
            text_doc.append(requestBody)
        text_doc = '\n'.join(text_doc)
        return text_doc
    
class Tools:

    def __init__(self, system, oas_spec):
        self.system = system
        api_spec = json.load(open(oas_spec))
        self.endpoint = {e['name']: Tool(e) for e in api_spec['endpoints']}
        self.host = api_spec['servers'][0]['url'] if 'servers' in api_spec else None
        self.headers = api_spec['headers']
    def match(self, name):
        return name

    def get_tool_list(self):
        tmp = [k for k, v in self.endpoint.items()]
        return tmp

    def formulate(self, tool, is_description=True, is_parameters=True, is_request_type=True, is_url=True,
                  execution_results_type=None, is_request_body=True):
        tool = self.match(tool)
        doc = self.endpoint[tool].formulate(is_description=is_description,
                                            is_parameters=is_parameters, is_url=is_url,
                                            execution_results_type=execution_results_type,
                                            is_request_type=is_request_type, is_request_body=is_request_body)
        return doc

class TMDBTools(Tools):

    def __init__(self, system, oas_spec):
        super(TMDBTools, self).__init__(system=system, oas_spec=oas_spec)

    def get_instruction(self, query, tools,
                        is_description=True,
                        is_parameters=True,
                        is_request_type=True,
                        execution_results_type='responses',
                        is_request_body=True,
                        is_url=True):
        docs = [f'{i}. ' + self.formulate(tool, is_description=is_description,
                                          is_parameters=is_parameters,
                                          is_request_body=is_request_body, is_url=is_url,
                                          execution_results_type=execution_results_type,
                                          is_request_type=is_request_type)
                for i, tool in enumerate(tools, start=1)]

        # instruction = GPT_TMDB_INSTRUCTION.format(system=self.system,headers=json.dumps(self.headers, indent=4), query=query, docs='\n\n'.join(docs))
        # instruction = TMDB_INST_STEP_BY.format(system=self.system,headers=json.dumps(self.headers, indent=4), query=query, docs='\n\n'.join(docs))
        instruction = CODETOOL_TMDB_INST.format(system=self.system,headers=json.dumps(self.headers, indent=4), query=query, docs='\n\n'.join(docs))

        return instruction
    

class RapidTools():

    def __init__(self):
        self.headers = {}

    def formulate(self, tool) -> str:
        text_doc = ["""API name: """ + tool['api_name']]

        text_doc.append('### API url\n' + config['stabletoolbench_server'])

        method = """### Request type\n""" + 'POST'
        text_doc.append(method)

        description = """### Description\n""" + tool['api_description'].strip()
        text_doc.append(description)

        parameters = '''### Parameter
- category(string, fixed): "{category}"
- tool_name(string, fixed): "{tool_name}"
- api_name(string, fixed): "{api_name}"
- tool_input(dict): {params}
- strip(string, fixed): ""
- toolbench_key(string, fixed): "{toolbench_key}"
'''.format(
        category=tool['category_name'], 
        tool_name=tool['tool_name'], 
        api_name=tool['api_name'],
        params=self.get_parameters(tool), 
        toolbench_key=config['toolbench_key']
    )

        text_doc.append(parameters)
        text_doc = '\n'.join(text_doc)

        return text_doc

    def get_parameters(self, api) -> str:
        if len(api['required_parameters']) == 0:
            return '{}'
        else:
            parameter = ['\n']
            for p in api['required_parameters']:
                if p['name'] == "id":
                    p['name'] = "is_id"
                parameter.append(f"  - {p['name'].lower()} ({p['type']}, required):")
                parameter.append(f'''    - Description: "{p['description']}"''')
                if isinstance(p['default'], str):
                    parameter.append(f'''    - Default: "{p['default']}"''') 
                else:               
                    parameter.append(f"    - Default: {p['default']}")
            for p in api['optional_parameters']:
                if p['name'] == "id":
                    p['name'] = "is_id"
                parameter.append(f"  - {p['name'].lower()} ({p['type']}, optional):")
                parameter.append(f'''    - Description: "{p['description']}"''')
                if isinstance(p['default'], str):
                    parameter.append(f'''    - Default: "{p['default']}"''') 
                else:               
                    parameter.append(f"    - Default: {p['default']}")

        return '\n'.join(parameter)

    def get_instruction(self, data):

        docs = [f'{i}. ' + self.formulate(tool)
                for i, tool in enumerate(data['api_list'], start=1)]
        
        instruction = INST.format(query=data['query'], docs='\n\n'.join(docs))

        return instruction


if __name__ == '__main__':
    # G1_instruction, G1_category, G1_tool, G2_instruction, G2_category, G3_instruction
    test_set = "G1_instruction"
    with open(f'./StableToolBench/solvable_queries/filter_test_instruction/{test_set}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(len(data))
    bar = tqdm(data[27:])
    rapidtool = RapidTools()

    for idx, line in enumerate(bar):
        instruction = rapidtool.get_instruction(line)
        print(instruction)
        break