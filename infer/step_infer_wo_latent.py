import json
import os
import random
import logging
import yaml
from tqdm import tqdm
from src.base import RapidTools
from infer.step_infer import get_openai_res, StepInfer

config = yaml.safe_load(open('infer/config.yaml', 'r'))

class StepInfer_wo_latent(StepInfer):
    def __init__(self):
        super().__init__()
    
    def select_sample(self, samples):
        cur_reward = []
        code_candidate = []
        exec_candidate = []
        for _, sample in enumerate(samples):
            reward, code, exec_res = self.get_cur_reward(sample)
            cur_reward.append(reward)
            code_candidate.append(code)
            exec_candidate.append(exec_res)

        
        indices = [i for i, value in enumerate(cur_reward) if value == 1]

        if indices:
            random_index = random.choice(indices)
        else:
            random_index = random.randint(0, len(cur_reward) - 1)

        best_index = random_index
        print(cur_reward)
        print(best_index)

        self.history_code_wo_print.append(self.replace_print_with_pass(code_candidate[best_index]))
        self.best_exec_res = exec_candidate[best_index]
        self.best_status = 0 if cur_reward[best_index] == 1 else 1

        return best_index, samples[best_index]
    

    def step_infer(self, instruction, query, all_paths):
        messages = [{"role": "user", "content": instruction}]
        path_name = "1"
        for step in range(self.MAX_DEPTH):
            logging.info(f'-----Step {step+1} generating...-----')
            stop_word = f"[Step {step+1} Finished]"
            samples = get_openai_res(messages, stop_word)
            best_index, best_sample = self.select_sample(samples)

            if self.history_ans == []:
                messages = messages + \
                    [{"role": "assistant", "content": best_sample + stop_word + '\n'}] + \
                    [{"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:2000], step=step+2)}]
            else:
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": ''.join(self.history_ans[:]) + best_sample + stop_word + '\n'},
                    {"role": "user", "content": self.next_user_query.format(info=self.best_exec_res[0:2000], step=step+2)}
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
    with open('StableToolBench/solvable_queries/filter_test_instruction/G2_category.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    bar = tqdm(data[0:5])

    for idx, item in enumerate(bar):
        save_path = f"data/stepwise_infer_result/gpt35_turbo_16k_wo_latent/G2_Category/{item['query_id']}.json"
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
            eval = StepInfer_wo_latent()
            result = eval.step_infer(instruction, item['query'], all_paths)

            logging.info(f"{item['query_id']}'s stepwise infer results is saved.")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        # break