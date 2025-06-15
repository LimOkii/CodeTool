Filter_Map = {
    # "G1_Instruction": [],
    "G1_Instruction": ['3510', '3922', '4505', '6736', '17223', '23248', '32807', '33330', '34823', '38494', '40054', '41806', '45775', '49991', '55323', '55489','58949', '61654', '67522', '68221', '70610', '75338', '72659', '77471', '79053', '80884', '81195', '81581', '82701', '83819', '85152', '86084'],
    "G1_Category": ['4286', '6504', '6511', '9719', '12688', '12770', '23486', '29647', '29653', '29778', '29917', '37847', '38028', '43375', '46662', '54640', '54801', '54839', '63151', '65185', '68448', '68553', '71583', '71801', '71823', '77171', '77261', '80298', '82434', '84974', '86231'],
    "G1_Tool": ['692', '2412', '2701', '3287', '7903', '7971', '8443', '9792', '9956', '14198', '16700', '17978', '19662', '25687', '26542', '26820', '28229', '28788', '32177', '33971', '34211', '34696', '38551', '43585', '44845', '45371', '45533', '48950', '53120', '65584', '66927', '69319', '69717', '70359', '76740', '77375', '77908', '85155', '85759'],
    "G2_Instruction": ['4746', '9957', '11820', '12034', '12507', '25866', '27543', '47748', '48770', '65468', '65521', '73783', '75958', '76230'],
    # "G2_Instruction": [],
    "G2_Category": ['3494', '3645', '3942', '4031', '4095', '13384', '13487', '13592', '14185', '14384', '29701', '33046', '33481', '42635', '42701', '42882', '42885', '43070', '50937', '71363', '72118', '72827', '79652', '79681'],
    "G3_Instruction": ['1983', '1984', '1985', '1989', '1991', '5863', '5864', '5865', '8032', '8334', '8335', '8337', '9341', '9343', '9344', '9345', '9346', '9349', '11644', '11645', '11647', '11648', '11649', '11650', '14485', '14489', '14938', '14950', '18978', '18979', '18980', '18982', '18984', '18987', '18988', '18990', '18992', '20022', '20024', '20026', '20027', '20028', '20029', '20030'],
    # "G3_Instruction": [],

}

eval_tool = {
    'tools': [
        {
            'type': 'function', 
            'function': {
                'name': 'check_answer_status', 
                'description': 'Parse the json answer with layerd nodes and return the answer_status about the answer', 
                'parameters': {
                    'type': 'object', 
                    'properties': {
                        'answer_status': {
                            'type': 'string', 
                            'enum': ['Unsure', 'Unsolved', 'Solved']
                            }, 
                        'reason': {
                            'type': 'string', 'description': 'explain your answer.'
                        }
                    }, 
                    'required': ['answer_status', 'reason']
                }
            }
        }
    ], 
    'tool_choice': {
        'type': 'function', 
        'function': {
            'name': 'check_answer_status'
            }
    }
}


EVAL_INST_V1 = '''Giving the query and answer, you need give `answer_status` of the answer by following rules:
1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

Query:
{query}
Answer:
{answer}

Now give your reason in "content" and `answer_status` of JSON to `check_answer_status`.'''

EVAL_INST_V2 = '''Giving the query and the answer, you need give `answer_status` of the answer by following rules:
1. If the answer doesn't contain any information that is helpful for answering the user's query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If the answer solve part of the query or not fully answer the query, return "Unsure".
2.3 If the answer is sufficient to solve the query, return "Solved".

Query:
{query}
Answer:
{answer}

Now give your reason in "content" and `answer_status` of JSON to `check_answer_status`.'''