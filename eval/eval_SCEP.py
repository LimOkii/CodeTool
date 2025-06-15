import os
import glob
import json

path_prefix = "/Users/luyifei/Projects/function_call/Final_infer/data/pass_rate"
# qwen25_coder_7b  qwen_wo_spot   qwen_wo_latent
models= ["qwen_wo_latent"]
test_sets = ["G1_Instruction", "G1_Category", "G1_Tool", "G2_Instruction", "G2_Category", "G3_Instruction"]

for model in models:
    for test_set in test_sets:
        total_sum = 0
        succ_sum = 0
        all_step_num = 0
        target_path = os.path.join(path_prefix, model, test_set)
        all_path = glob.glob(os.path.join(target_path, '*.json'))
        for file in all_path:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in data['exec_res'].keys():
                if data['exec_res'][key]['status'] == 0:
                    succ_sum += 1
            total_sum += len(data['exec_res'].keys())
            all_step_num += total_sum
            # print(data['exec_res'].keys())
        # print(len(all_path))
        print(succ_sum, total_sum)
        print(succ_sum / total_sum)
        
        # print(succ_sum)
        # break
    print(all_step_num)
    break