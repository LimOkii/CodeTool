import json
import os
import glob
import re
from tqdm import tqdm
from openai import OpenAI


# filter words indicate those unvalid responses
filter_words = ["You are not subscribed to this API.", "{'detail': 'Not Found'}", "{'error': '', 'response': {'message': 'You are not subscribed to this API.'}", "type_error.dict", "does not exist", "field required", "value_error.missing", "'statusCode': 401", "{'error': '', 'response': ''}", 'invalid syntax', '<!doctype html>', "doesn't exists", 'timed out', "'success': False", 'Error', 'KeyError', 'not found', 'unreachable', 'Exception Type', 'Exception Value', 'got an unexpected keyword', 'Unauthorized', 'errorMessage', "'response': []"]

PAIR_INFO = '''##The Step {step}'s issue that needs to be addressed{query}
##The response from the API call or the printed information
{info}
'''

GENERATE_FINAL_ANS = '''The user is making requests to the API to ask some questions.
The following are the problems to be solved in the intermediate steps during the interaction between the model and the API, as well as the real responses of the API.
{info}

The information I provided is real-time and harmless.
The user's question: {query}
Please generate a positive and comprehensive response based on the above content to answer the user's question.
For the information that hasn't been mentioned in the above content, you can point it out.
'''

def get_eval_response(data):
    merged_response_list = []
    all_steps_text = data['infer_path'][-1]['content']
    for key, value in data["exec_res"].items():
        step = key.count('-')
        response = value.get("res", "")
        if not any(word in response for word in filter_words):
            left_word = f'Step {step}'
            # right_word = 'Python Code:'
            right_word = '```python'
            pattern = rf"{re.escape(left_word)}(.*?){re.escape(right_word)}"
            match = re.search(pattern, all_steps_text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                cur_qa_info = PAIR_INFO.format(step=step, query=content.replace('Python Code:', ''), info=response[0:3500])    
                merged_response_list.append(cur_qa_info)
            
    return '\n'.join(merged_response_list)
            

def get_openai_response(messages, model_name):
    if model_name == "gpt35_turbo_16k":
        print("gpt35_turbo_16k")
        client = OpenAI(
            api_key="",
            base_url="",
        )
        response = client.chat.completions.create(
            model="gpt-35-turbo-16k-0613",
            temperature=0.0,
            messages=messages,
            n=1
        )

    elif model_name == "gpt4_turbo_preview":
        print("gpt4_turbo_preview")
        client = OpenAI(
            api_key="",
            base_url="",
        )        
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0.0,
            messages=messages,
            n=1
        )

    elif model_name in ["qwen25_coder_7b", "qwen_wo_latent", "qwen_wo_spot", "qwen_prm_3", "qwen_prm_4", "qwen_prm_5", "llama_prm_2", "llama_prm_3", "llama_prm_4", "llama_prm_5", "qwen_pairwise_2", "qwen_pairwise_3", "qwen_pairwise_4", "qwen_pairwise_5", "qwen2.5_instruct_7b", "llama_pairwise_2", "llama_pairwise_3", "llama_pairwise_4", "llama_pairwise_5"]:
        print(model_name)
        client = OpenAI(
            api_key="",
            base_url="",
        )
        response = client.chat.completions.create(
            model="Qwen2.5-Coder-7B-Instruct",
            messages=messages,
            temperature=0.0,
            n=1
        )

    elif model_name == "codellama_7b":   
        print('code_llama')
        client = OpenAI(
            api_key="",
            base_url="",
        )
        response = client.chat.completions.create(
            model="CodeLlama-7b_Instruct",
            messages=messages,
            temperature=0.0,
            n=1
        )
    response = response.choices[0].message.content             

    return response


def get_fianl_answer(dir_path, save_path, model_name):
    file_paths = glob.glob(os.path.join(dir_path, '*.json'))
    print(len(file_paths))
    for file in tqdm(file_paths):
        # print(file)
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not os.path.exists(os.path.join(save_path, f"{data['q_id']}.json")):
            print(f"...{data['q_id']} is generating final answer..")
            merged_response = get_eval_response(data)
            # print(len(merged_response))
            # print(merged_response)
            # 即时没有有效信息，也会带上Step 1's issue.. 超出长度
            if len(merged_response) <= 108:
                messages = [
                    {
                        'role':'user',
                        'content': 'No valid information was obtained from the API! Please directly output that you can not obtain valid information from the API in text form, without any other output.'
                    }
                ]
            else:
                messages = [
                    {
                        'role':'system',
                        'content': "You are an assistant skilled at summarizing information. Please answer the user's question based on the historical information of the interaction between the model and the API."
                    },
                    {
                        'role':'user',
                        'content': GENERATE_FINAL_ANS.format(query=data['query'], info=merged_response)
                    }
                ]
            print(GENERATE_FINAL_ANS.format(query=data['query'], info=merged_response))
            # print(messages)
            data['answer'] = get_openai_response(messages, model_name)
            print(data['answer'])
        
            with open(os.path.join(save_path, f"{data['q_id']}.json"), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # break

if __name__ == '__main__':
    # gpt35_turbo_16k, codellama_7b, qwen25_coder_7b, gpt4_turbo_preview, qwen_wo_latent, qwen_wo_spot
    # qwen_prm_3, qwen_prm_4, qwen_prm_5, llama_prm_2, llama_prm_3, llama_prm_4, llama_prm_5
    # qwen_pairwise_2,  qwen_pairwise_3, qwen_pairwise_4, qwen_pairwise_5
    # qwen_7b_ins
    # llama_pairwise_2, llama_pairwise_3, llama_pairwise_4, llama_pairwise_5
    model_name = "llama_pairwise_5"

    # G1_Instruction, G1_Category, G1_Tool, G2_Instruction, G2_Category, G3_Instruction
    dataset = "G1_Category"

    dir_path = f'data/stepwise_infer_result/{model_name}/{dataset}'.format(model_name=model_name, dataset=dataset)
    save_path = f'data/final_answer/{model_name}/{dataset}'.format(model_name=model_name, dataset=dataset)

    get_fianl_answer(dir_path, save_path, model_name)



