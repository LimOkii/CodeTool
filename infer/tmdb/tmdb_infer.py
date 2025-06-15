import os, re, copy
import json, yaml
import requests
import logging
from tqdm import tqdm
from copy import deepcopy
from src.base import TMDBTools
from src.engine import PythonExecNet
from src.instruction import *
from infer.step_infer import StepInfer
from infer.step_infer import get_openai_res

config = yaml.safe_load(open('infer/config.yaml', 'r'))


class StepInfer_TMDB(StepInfer):
    def __init__(self, headers):
        super().__init__()
        self.headers = headers
        self.exec = PythonExecNet(self.headers)
        self.next_user_query = '''##The result of the url requests:
```json
{info}
```
Please note that the responses I provide you with may not be complete, but they cover the field names that you need to parse.
Continue generating python code according to the previous code and the result of the url requests.
#When you start generating code, please strictly follow the points below:
##1. If the response field is in the form of a list or a dictionary, use python code format to parse the required fields from result of the url requests which will be used for the next URL request.
##2. If the response field is in the form of a string, please obtain the information you need from it.
##3. When you request a new URL, print the JSON data of it. 
##4. If I haven't provided the API response results, please do not use your own knowledge to parse the fields!
##5. The parameters for requesting the API must be consistent with the given parameter names.If the API is not available or you don't receive useful information, try to call another api.
# You only need to generate the code for the next step. Do not generate code for multiple steps.
# When you think you have obtained the final answer to query: **{query}**, print a complete final answer and end with '[All Finished]'. 
# Otherwise, generate you response strictly begin with '##Step {step}', end with '[Step {step} Finished]'. 
'''

    def step_infer(self, instruction, query, all_paths):
        messages = [{"role": "user", "content": instruction}]
        path_name = "1"
        # for step in range(1):
        for step in range(self.MAX_DEPTH):
            print(f'-----Step {step+1} generating...-----')
            stop_word = f"[Step {step+1} Finished]"
            stop_word = f"[Step {step+1} Finished]"
            samples = get_openai_res(messages, stop_word)
            print(samples)
            best_index, best_sample = self.select_sample(instruction, query, samples, step)

            if self.history_ans == []:
                messages = messages + \
                    [{"role": "assistant", "content": best_sample + stop_word + '\n'}] + \
                    [{"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:1500], step=step+2, query=query)}]
            else:
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": ''.join(self.history_ans[:]) + best_sample + stop_word + '\n'},
                    {"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:1500], step=step+2, query=query)}
                ]

            self.history_ans.append(best_sample + stop_word + '\n')

            cur_path_name = f"{path_name}-{best_index + 1}"
            path_name = cur_path_name

            all_paths['infer_path'].append({
                    "path_name": cur_path_name,
                    "content": ''.join(self.history_ans[:]),
                    "is_leaf": '[All Finished]' in best_sample or len(self.history_ans) == self.MAX_DEPTH
                })
            
            all_paths['exec_res'].setdefault(cur_path_name, {})['code'] = '\n'.join(self.history_code_wo_print[:])
            all_paths['exec_res'].setdefault(cur_path_name, {})['res'] = self.best_exec_res[0:1500]
            all_paths['exec_res'].setdefault(cur_path_name, {})['status'] = self.best_status

            
            if '[All Finished]' in best_sample:
                break

        return all_paths


if __name__ == '__main__':
    with open('infer/tmdb/tmdb.data.candidate=20.v2.json', 'r') as f:
        data = json.load(f)

    toolsets = TMDBTools(
        system='''Here are some APIs used to access the TMDB platform. You need to answer the question step by step based on the appropriate APIs, provide Python code for each step and print the final answer. The API can be accessed via HTTP request.''',
        oas_spec='infer/tmdb/tmdb.spec.raw.v2.json',
    )
    
    print(len(data))
    print(data[0])
    bar = tqdm(data)

    for idx, item in enumerate(bar):
        save_path = f"data/stepwise_infer_result/tmdb/{item['qid']}.json"
        if not os.path.exists(save_path):
            print('------',idx+1,'coding -------')
            print(item['qid'])
            tools = copy.deepcopy(item['api_list'])
            instruction = toolsets.get_instruction(item['query'], tools)
            print(instruction)
            all_paths = {
                "q_id": item['qid'],
                "query": item['query'],
                "instruction": instruction,
                "infer_path": [],
                "exec_res": {},
            }
            eval = StepInfer_TMDB(headers=config['tmdb_headers'])
            result = eval.step_infer(instruction, item['query'], all_paths)

            print('save')
            logging.info(f"{item['qid']}'s stepwise infer results is saved.")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        break