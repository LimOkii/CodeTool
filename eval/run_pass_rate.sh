export test_set=G1_Category # G1_Instruction, G1_Category, G1_Tool, G2_Category, G2_Instruction, G3_Instruction
MODEL_NAME=llama_pairwise_4
# gpt35_turbo_16k, qwen25_coder_7b, codellama_7b, qwen_wo_latent, qwen_wo_spot
# qwen_prm_3, qwen_prm_4, qwen_prm_5, llama_prm_2, llama_prm_3, llama_prm_4, llama_prm_5
# qwen_pairwise_2, qwen_pairwise_3, qwen_pairwise_4, qwen_pairwise_5
# llama_pairwise_2, llama_pairwise_3, llama_pairwise_4, llama_pairwise_5

python eval/eval_pass_rate.py \
    --data_path_prefix data/final_answer/${MODEL_NAME} \
    --save_path_prefix data/pass_rate/${MODEL_NAME} \
    --inst_version v2 \
    --test_set ${test_set} \

