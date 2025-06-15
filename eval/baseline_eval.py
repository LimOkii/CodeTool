import os
import json
import glob
import yaml
import argparse
from tqdm import tqdm
from openai import OpenAI
from eval.utils import *

config = yaml.safe_load(open('eval/config.yaml', 'r'))
print(config)

client = OpenAI(
    api_key = config['api_key'],
    base_url = config['api_base'],
)

def get_openai_res(text):
    response = client.chat.completions.create(
        model = config['model'],
        temperature = config['temperature'],
        messages = [
            {
                'role': 'user',
                'content': text,
            }
        ],
        timeout = 600,
        **eval_tool
    )
    print(response)
    response = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return response

def eval(file_path, save_path, inst_version):
    data = json.load(open(file_path, 'r'))
    for item in tqdm(data):
        if not os.path.exists(os.path.join(save_path, f"{item['q_id']}.json")):
            save_format = {
                "q_id": item['q_id'],
                "query": item['query'],
            }
            INST = EVAL_INST_V1 if inst_version == 'v1' else EVAL_INST_V2
            text = INST.format(query=item['query'], answer=item['answer'])
            response = get_openai_res(text)
            save_format['final_answer'] = item['answer']
            save_format['reason'] = response['reason']
            save_format['answer_status'] = response['answer_status']
            with open(os.path.join(save_path, f"{item['q_id']}.json"), 'w', encoding='utf-8') as f:
                json.dump(save_format, f, ensure_ascii=False, indent=4)
        # break

def calculate_pass_rate(file_path, test_set):
    all_paths = glob.glob(os.path.join(file_path, '*.json'))
    filter_ids = Filter_Map[test_set]
    sum = 0
    for file in tqdm(all_paths):
        if any(id in file for id in filter_ids):
            continue
        data = json.load(open(file, 'r'))
        if data['answer_status'].lower() == "unsure":
            sum += 0.5
        elif data['answer_status'].lower() == "unsolved":
            sum += 0.0
        else:
            sum += 1
    if test_set == "G1_Instruction":
        valid_length = 163 - len(filter_ids) 
    else:
        valid_length = len(all_paths) - len(filter_ids)
        # valid_length = 158 - len(filter_ids)

    # print(f"The number of valid test set for {test_set} is:", valid_length)
    print(f"The total score of valid test set for {test_set} is:", sum)
    print(f"The solvable pass rate for {test_set} is:", sum / len(all_paths))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # StableToolBench给的推理结果转换完成的简单形式
    parser.add_argument('--data_path', type=str, default='/data/baselines/gpt-4-turbo-preview_dfs/G1_Instruction/gpt-4-turbo-preview_dfs_G1_Ins.json')
    # 调接口得到pass_rate的评估结果  根据不同的评估prompt 放在 v1 or v2下
    parser.add_argument('--save_path_prefix', type=str, default='/data/baselines/gpt-4-turbo-preview_dfs')
    # 要用哪一版的评估prompt
    parser.add_argument('--inst_version', type=str, default='v1', help='which eval instruction to use')
    # 选择测试集
    parser.add_argument('--test_set', type=str, default='G1_Instruction', help='which test set to use')

    args = parser.parse_args()
    save_path = os.path.join(args.save_path_prefix, args.test_set, args.inst_version)

    print(args.data_path)
    print(save_path)
    eval(args.data_path, save_path, args.inst_version)
    calculate_pass_rate(save_path, args.test_set)