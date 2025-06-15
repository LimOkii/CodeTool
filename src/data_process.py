import re
import yaml
from tqdm import tqdm
from openai import OpenAI
from engine import PythonExecNet

config = yaml.safe_load(open('src/config.yaml', 'r'))

class DataProcess:
    def __init__(self, all_paths, progress_bar):
        self.MAX_DEPTH = config['MAX_DEPTH']
        self.history_ans = []
        self.history_code_wo_print = []
        self.all_paths = all_paths
        self.pattern = r"```python(.*?)```"
        self.progress_bar = progress_bar
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
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
**Ensure the code blocks are correctly started and ended with three backticks (```) for step {step}**
# Now generate you response strictly begin with '##Step {step}'!
'''
    def replace_print_with_pass(self, code_info):
        pattern = r'print\(([^()]*(?:\([^()]*\))*[^()]*)\)'
        code_info = re.sub(pattern, 'pass', code_info, flags=re.DOTALL)
        return code_info
    
    def get_openai_res(self, messages, stop_word):
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
            messages = messages,
            stop = [stop_word],
            timeout = 600,
            )
        
        responses = [response.choices[i].message.content for i in range(config['n'])]

        return responses
    

    def recursive_generation(self, step=1, path_name="1", messages=None):
        if messages is None:
            messages = [{"role": "user", "content": self.all_paths['instruction']}]

        if len(self.history_ans) > 0 and '[All Finished]' in self.history_ans[-1]:
            return
        
        if len(self.history_ans) >= self.MAX_DEPTH:
            return
        
        # if len(self.history_ans) > 0 and '[Step 2 Finished]' in self.history_ans[-1]:
        #     return
        
        stop_word = f"[Step {step} Finished]"        
        samples = self.get_openai_res(messages, stop_word)

        for i, sample in enumerate(samples):
            # print(sample)
            # 当前步骤的python代码
            py_code = re.findall(self.pattern, sample, re.DOTALL)
            if len(py_code) != 0:
                if self.history_code_wo_print == []:
                    cur_pycode = '\n'.join(py_code).strip()
                    exec_res = self.exec.run(cur_pycode)
                else:
                    cur_pycode = '\n'.join(self.history_code_wo_print[:]) + '\n' + py_code[0]
                    exec_res = self.exec.run(cur_pycode.strip())
            
                self.history_code_wo_print.append(self.replace_print_with_pass(py_code[0]))
            

                if self.history_ans == []:
                    next_messages = messages + \
                        [{"role": "assistant", "content": sample + stop_word + '\n'}] + \
                        [{"role": "user", "content": self.next_user_query.format(info=exec_res[0], step=step+1)}]
                else:
                    next_messages = [
                        {"role": "user", "content": self.all_paths['instruction']},
                        {"role": "assistant", "content": ''.join(self.history_ans[:]) + sample + stop_word + '\n'},
                        {"role": "user", "content": self.next_user_query.format(info=exec_res[0], step=step+1)}
                    ]

                self.history_ans.append(sample + stop_word + '\n')

                cur_path_name = f"{path_name}-{i + 1}"

                self.all_paths['infer_path'].append({
                    "path_name": cur_path_name,
                    "content": ''.join(self.history_ans[:]),
                    "is_leaf": '[All Finished]' in sample or len(self.history_ans) == self.MAX_DEPTH
                })
                
                self.all_paths['exec_res'].setdefault(cur_path_name, {})['code'] = '\n'.join(self.history_code_wo_print[:])
                self.all_paths['exec_res'].setdefault(cur_path_name, {})['res'] = exec_res[0]
                self.all_paths['exec_res'].setdefault(cur_path_name, {})['status'] = exec_res[1]

                self.progress_bar.update(1)
                             
            else:
                cur_path_name = f"{path_name}-{i + 1}"

                self.all_paths['infer_path'].append({
                    "path_name": cur_path_name,
                    "content": ''.join(self.history_ans[:]),
                    "is_leaf": True
                })

                self.all_paths['exec_res'].setdefault(cur_path_name, {})['code'] = '\n'.join(self.history_code_wo_print[:])
                self.all_paths['exec_res'].setdefault(cur_path_name, {})['res'] = sample
                self.all_paths['exec_res'].setdefault(cur_path_name, {})['status'] = 0

                if self.history_ans:
                    self.history_ans.pop()
                    
                if self.history_code_wo_print:    
                    self.history_code_wo_print.pop()

                return
      
            if '[All Finished]' not in sample:
                self.recursive_generation(step + 1, cur_path_name, next_messages)

            if self.history_ans:
                self.history_ans.pop()

            if self.history_code_wo_print:
                self.history_code_wo_print.pop()

        return