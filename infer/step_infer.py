import json
import re
import os
import requests
from tqdm import tqdm
import logging
import yaml
from openai import OpenAI
from src.base import RapidTools
from src.engine import PythonExecNet
from src.instruction import REWARD_MODEL_PROMPT_V2

config = yaml.safe_load(open('infer/config.yaml', 'r'))
logging.basicConfig(level=logging.INFO)

def get_openai_res(messages, stop_word):
    # print(config)
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['api_base'],
    )
    response = client.chat.completions.create(
        model = config['model'],
        temperature = config['temperature'],
        top_p = config['top_p'],
        n = config['n'],
        max_tokens=config['max_tokens'],
        messages = messages,
        stop = [stop_word],
        timeout = 600,
    )
    
    responses = [response.choices[i].message.content for i in range(config['n'])]

    return responses


class StepInfer:
    def __init__(self):
        self.MAX_DEPTH = 5
        self.history_ans = []
        self.best_exec_res = ''
        self.best_status = 0
        self.history_code_wo_print = []
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.exec = PythonExecNet(self.headers)
        self.rm_url = config['rm_url']
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
**Ensure the code blocks are correctly started and ended with three backticks (```) for step {step}**
# You only need to generate the code for the next step. Do not generate code for multiple steps.
# Now generate you response strictly begin with '##Step {step}', end with [Step {step} Finished]
'''
        self.his_info_prompt = '''##The previous code:
{previous_code}
##The result of the url requests:
```json
{info}
```
'''
    def replace_print_with_pass(self, code_info):
        pattern = r'print\(([^()]*(?:\([^()]*\))*[^()]*)\)'
        code_info = re.sub(pattern, 'pass', code_info, flags=re.DOTALL)
        return code_info
    

    def get_cur_reward(self, sample):
        pattern = r"```python(.*?)```"
        py_code = re.findall(pattern, sample, re.DOTALL)

        if len(py_code) != 0:
            if self.history_code_wo_print == []:
                cur_pycode = '\n'.join(py_code).strip()
                exec_res = self.exec.run(cur_pycode)
            else:
                cur_pycode = '\n'.join(self.history_code_wo_print[:]) + '\n' + py_code[0]
                exec_res = self.exec.run(cur_pycode.strip())
        
            cur_reward = 1 if exec_res[1] == 0 else 0

            # reward, py_code, exec_res
            return cur_reward, py_code[0], exec_res[0]
        else:
            cur_reward = 0
            return cur_reward, '', ''
    
    def select_sample(self, instruction, query, samples, step):
        rm_system_info = instruction.split('Based on provided APIs')[0]
        stop_word = f"[Step {step+1} Finished]"
        cur_reward = []
        estimate_reward = []
        code_candidate = []
        exec_candidate = []
        for i, sample in enumerate(samples):
            if step == 0:
                rm_messages = [
                    {"role": "system", "content": REWARD_MODEL_PROMPT_V2.format(system_info=rm_system_info, user_query=query, his_info="")},
                    {"role": "user", "content": sample + stop_word + '\n'}
                ]
            else:
                his_info = self.his_info_prompt.format(previous_code='/n'.join(self.history_ans[:]), info=self.best_exec_res[0:2000])
                rm_messages = [
                    {"role": "system", "content": REWARD_MODEL_PROMPT_V2.format(system_info=rm_system_info, user_query=query, his_info=his_info)},
                    {"role": "user", "content": sample + stop_word + '\n'}
                ]

            response = requests.post(
                self.rm_url, 
                json={
                    "index": i+1,
                    "messages": rm_messages
                }
            )
            estimate_reward.append(response.json()['response'])
            reward, code, exec_res = self.get_cur_reward(sample)
            print(reward)
            cur_reward.append(reward)
            code_candidate.append(code)
            exec_candidate.append(exec_res)
            
        best_index = 0 if cur_reward[0] + estimate_reward[0] >= cur_reward[1] + estimate_reward[1] else 1
        # print(f"---Step {step+1}----'s reward")
        # print(cur_reward)
        self.history_code_wo_print.append(self.replace_print_with_pass(code_candidate[best_index]))
        self.best_exec_res = exec_candidate[best_index]
        self.best_status = 0 if cur_reward[best_index] == 1 else 1

        return best_index, samples[best_index]


    def step_infer(self, instruction, query, all_paths):
        messages = [{"role": "user", "content": instruction}]
        path_name = "1"
        for step in range(self.MAX_DEPTH):
            logging.info(f'-----Step {step+1} generating...-----')
            # print(f'-----Step {step+1} generating...-----')
            stop_word = f"[Step {step+1} Finished]"
            samples = get_openai_res(messages, stop_word)
            best_index, best_sample = self.select_sample(instruction, query, samples, step)

            if self.history_ans == []:
                messages = messages + \
                    [{"role": "assistant", "content": best_sample + stop_word + '\n'}] + \
                    [{"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:500], step=step+2)}]
            else:
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": ''.join(self.history_ans[:]) + best_sample + stop_word + '\n'},
                    {"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:500], step=step+2)}
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
            all_paths['exec_res'].setdefault(cur_path_name, {})['res'] = self.best_exec_res
            all_paths['exec_res'].setdefault(cur_path_name, {})['status'] = self.best_status
            
            if '[All Finished]' in best_sample:
                break

        return all_paths


if __name__ == '__main__':
    with open('StableToolBench/solvable_queries/filter_test_instruction/G1_instruction.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    bar = tqdm(data)
    print(len(data))

    for idx, item in enumerate(bar):
        # G2_Ins
        # if idx == 64 or idx == 69:
        #     continue
        # G1_Tool
        # if idx == 33:
        #     continue

        save_path = f"data/stepwise_infer_result/qwen2_7b/G1_Instruction/{item['query_id']}.json"
        if not os.path.exists(save_path):
            print('------',idx+1,'coding -------')
            print(item['query_id'])
            instruction = RapidTools().get_instruction(item)
            # print(instruction)
            all_paths = {
                "q_id": item['query_id'],
                "query": item['query'],
                "instruction": instruction,
                "infer_path": [],
                "exec_res": {},
            }
            eval = StepInfer()
            result = eval.step_infer(instruction, item['query'], all_paths)
            
            logging.info(f"{item['query_id']}'s stepwise infer results is saved.")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        # break