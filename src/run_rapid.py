import os
import json
import time
import glob
import argparse
import logging
from tqdm import tqdm
from instruction import *
from base import RapidTools
from data_process import DataProcess

logging.basicConfig(level=logging.INFO)

def run_rapid(data_file, save_path_prefix):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    toolsets = RapidTools()

    bar = tqdm(data[0:1])
    # bar = tqdm(data)

    for item in bar:
        save_path = f"{save_path_prefix}/{item['query_id']}_path.json"
        if not os.path.exists(save_path):
            instruction = toolsets.get_instruction(item)
            # print(item['query_id'], "is generating...")
            logging.info(f"{item['query_id']} is generating...")
            all_paths = {
                "q_id": item['query_id'],
                "query": item['query'],
                "instruction": instruction,
                "infer_path": [],
                "exec_res": {},
            }

            start_time = time.time()

            progress_bar = tqdm(total=80, desc="Processing", unit="sample")
            process = DataProcess(all_paths, progress_bar)
            process.recursive_generation()

            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(all_paths, file, ensure_ascii=False, indent=4)

            logging.info(f"{item['query_id']}_infer_execution time: {time.time()-start_time} seconds")


def get_scored_data_v1(dir_prefix, save_path):
    all_files = glob.glob(f"{dir_prefix}/*.json")
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = score_path(data)
        with open(os.path.join(save_path, f"{data['q_id']}_scored.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        # break

    print('The num of total train data:', len(all_files))    

    return 


def get_pair_data(scored_path, pair_path):
    all_files = glob.glob(f"{scored_path}/*.json")
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        res = []
        for idx, path in enumerate(data['infer_path']):
            if path['path_name'][-1] == '2':
                continue
            target_path_name = path['path_name'][:-1] + '2'
            
            for target_path in data['infer_path']:
                if target_path['path_name'] == target_path_name:
                    if path['current_exec'] != target_path['current_exec']:
                        res.append([path, target_path])
                        break
                    elif target_path['hard_estimate'] != path['hard_estimate'] or target_path['latent_reward'] != path['latent_reward']:
                        res.append([path, target_path])
                        break
        
        selected_train_data = {
            "q_id": data['q_id'],
            "query": data['query'],
            "instruction": data['instruction'],
            "pair_path": res,
            "exec_res": data['exec_res']
        }
        
        with open(os.path.join(pair_path, f"{data['q_id']}_pair.json"), 'w', encoding='utf-8') as f:
            json.dump(selected_train_data, f, ensure_ascii=False, indent=4)
        
        # break
    return

def score_path(data):
    # 所有的叶子节点
    leave_node = [[each_path['path_name'], each_path['content']] for each_path in data['infer_path'] if each_path['is_leaf']]
    # print(leave_node[0][0])
    for cur_path in tqdm(data['infer_path']):
        hard_flag = False
        # 首先检查当前节点是否可运行
        if data['exec_res'][cur_path['path_name']]['status'] == 0:
            cur_path['current_exec'] = True
        else:
            cur_path['current_exec'] = False
        
        # 当前节点到其所有叶子节点的路径
        cur_path_to_leaf = [s[0] for s in leave_node if s[0].startswith(cur_path['path_name'])]

        # 当前节点到所有其叶子结点的路径数和 = 路径的'-'总数 减去 （当前路径'-'数）* 叶子节点数
        tau = sum(roll_out.count('-') for roll_out in cur_path_to_leaf) - cur_path['path_name'].count('-') * len(cur_path_to_leaf)


        path_sum = 0
        for leaf_name in cur_path_to_leaf:
            if data['exec_res'][leaf_name]['status'] == 0:
                path_sum += 1
                hard_flag = True

        if len(cur_path_to_leaf) > 0:
            total_path_num = len(cur_path_to_leaf)
            cur_path['soft_estimate'] = path_sum / total_path_num
            cur_path['hard_estimate'] = 1 if hard_flag == True else 0

        alpha = 0.9
        beta = 0.95
        L = 20
        latent_reward = alpha ** (1-cur_path['soft_estimate']) * (beta ** (tau/L))
        cur_path['latent_reward'] = latent_reward

        # break

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # G1_Instruction, G1_Category, G1_Tool, G2_Category, G2_Instruction, G3_Instruction
    # test_set = 'G1_Instruction'
    # download from ToolBench    
    parser.add_argument('--train_candidate_file', type=str, default='data/instruction/G1_query.json')

    parser.add_argument('--infer_path_save', type=str, default='data/process_data/G1_Instruction/dfs_data')

    parser.add_argument('--scored_path_save', type=str, default='data/process_data/G1_Instruction/path_scored')
    
    parser.add_argument('--pair_path_save', type=str, default='data/process_data/G1_Instruction/pair_path')

    args = parser.parse_args()

    # generate the infer_path data from gpt --> data/{test_set}/dfs_data/q**_path.json
    run_rapid(args.train_candidate_file, args.infer_path_save)

    # score each infer path --> data/{test_set}/path_scored/q**_scored.json
    get_scored_data_v1(args.infer_path_save, args.scored_path_save)

    # from scored path to data_pair path
    get_pair_data(args.scored_path_save, args.pair_path_save)

