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
        messages = [{
            'role':'user',
            'content': text,
        }],
        timeout = 600,
        **eval_tool
        )
    print(response.choices[0])
    response = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return response

def eval(file_path, save_path, inst_version):
    file_paths = glob.glob(os.path.join(file_path, '*.json'))
    for file_path in tqdm(file_paths):
        item = json.load(open(file_path, 'r'))
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

    # valid_length = len(all_paths) - len(filter_ids) 

    # print(f"The number of valid test set for {test_set} is:", valid_length)
    print(f"The total score of valid test set for {test_set} is:", sum)
    print(f"The solvable pass rate for {test_set} is:", sum / len(all_paths))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # PRM + Infer 结束之后的文件   放在 data/test_set_infer_result + 具体日期下
    parser.add_argument('--data_path_prefix', type=str, default='data/test_set_infer_result/1212')
    # 再次调接口得到final_answer的文件保存路径  放在./data/eval_result/1212 + test_set + 具体评估版本下
    parser.add_argument('--save_path_prefix', type=str, default='data/eval_result/1212')
    # 要用哪一版的评估prompt
    parser.add_argument('--inst_version', type=str, default='v1', help='which eval instruction to use')
    # 选择测试集
    parser.add_argument('--test_set', type=str, default='G1_Instruction', help='which test set to use')

    args = parser.parse_args()

    data_path = os.path.join(args.data_path_prefix, args.test_set)
    save_path = os.path.join(args.save_path_prefix, args.test_set, args.inst_version)
    # print(data_path)
    # print(save_path)
    # print(glob.glob(os.path.join(save_path, '*.json')))
    eval(data_path, save_path, args.inst_version)
    calculate_pass_rate(save_path, args.test_set)
