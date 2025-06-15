import json
import glob
import os


REWARD_MODEL_PROMPT = '''
# Task Description
The user is writing the code for API requests and parsing API responses to complete a query.
Your task is to evaluate the rationality of the code given by the user at each step.

# API Information
{system_info}
{his_info}
# You need to consider the following factors:
1. The API request made by the user at the current step is a necessary step to complete the query.
2. The names of the parameters for requesting the API should be strictly in line with the specified required_parameter.
3. The user needs to parse the content on the fields of the given response and should not create unknown field names.
4. The contribution of the user's API request at the current step to the completion of the query.

# The query is: 
"{user_query}"

# If you think the code written by the user meets the above requirements, output "yes". Otherwise, output "no".
'''


def get_sft_data(data):
    res = []

    for item in data['pair_path']:
        step = item[0]['path_name'].count("-")
        system_info = data['instruction'].split('Based on provided APIs')[0]
        if step == 1:
            for i in range(2):  # 0 和 1
                system = REWARD_MODEL_PROMPT.format(system_info=system_info, user_query=data['query'], his_info="").strip()
                if item[i]['current_exec'] == item[1-i]['current_exec']:
                    target_format = {
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": item[i]['content']
                                },
                                {
                                    "from": "gpt",
                                    "value": 'yes' if item[i]['hard_estimate'] > item[1 - i]['hard_estimate'] or \
                                        item[i]['soft_estimate'] > item[1 - i]['soft_estimate'] else 'no'
                                }
                            ],
                            "system": system,
                    }
                else:
                    target_format = {
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": item[i]['content']
                                },
                                {
                                    "from": "gpt",
                                    "value": 'yes' if item[i]['current_exec'] == True else 'no'
                                }
                            ],
                            "system": system,
                    }

                res.append(target_format)
        else:
            for i in range(2):
                split_word = f"[Step {step-1} Finished]"
                previous_code = item[i]['content'].split(split_word)[0] + split_word
                exec_name = item[i]['path_name'][0:-2]
                exec_info = data['exec_res'][exec_name]['res']

                his_info = '''
# The previous code provided by the user:
{previous_code}
# The result of the url requests:
```json
{info}

```
'''.format(previous_code=previous_code, info=exec_info)
                
                system = REWARD_MODEL_PROMPT.format(system_info=system_info, user_query=data['query'], his_info=his_info).strip()

                if item[i]['current_exec'] == item[1-i]['current_exec']:
                    if split_word in item[i]['content']:
                        target_format = {
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": item[i]['content'].split(split_word)[1]
                                    },
                                    {
                                        "from": "gpt",
                                        "value": 'yes' if item[i]['hard_estimate'] > item[1 - i]['hard_estimate'] or \
                                            item[i]['soft_estimate'] > item[1 - i]['soft_estimate'] else 'no'
                                    }
                                ],
                                "system": system,
                            }
                        res.append(target_format)
                else:
                    if split_word in item[i]['content']:
                        target_format = {
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": item[i]['content'].split(split_word)[1]
                                    },
                                    {
                                        "from": "gpt",
                                        "value": 'yes' if item[i]['current_exec'] == True else 'no'
                                    }
                                ],
                                "system": system,
                            }
                        res.append(target_format)
        # print(step)
        # break

    return res


if __name__ == '__main__':
    dir_prefix = 'data/process_data/G1_Instruction/pair_path'
    sft_data_path = 'data/rm_train_data'
    
    final_res = []
    all_files = glob.glob(f"{dir_prefix}/*.json")
    for file in all_files:
        print(file)
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        final_res.extend(get_sft_data(data))
        # break

    with open(os.path.join(sft_data_path, "G1_Inst_0303_v1.json"), 'w', encoding='utf-8') as f:
        json.dump(final_res, f, ensure_ascii=False, indent=4)

