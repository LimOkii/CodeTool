export model_name=qwen25 # toolllama  gpt-4-turbo-preview  gpt-35-turbo-16k-0613  toolllama_steptool, qwen25
export method=dfs  # dfs, cot
export version=v2 # v1, v2
export test_set=G1_Category # G1_Instruction, G1_Category, G1_Tool, G2_Category, G2_Instruction, G3_Instruction
strategy=${model_name}_${method}

mkdir -p data/baselines/${strategy}/${test_set}/${version}

python eval/baseline_eval.py \
    --data_path data/baselines/${strategy}/${test_set}/${strategy}_${test_set}.json \
    --save_path_prefix data/baselines/${strategy} \
    --inst_version ${version} \
    --test_set ${test_set} \

