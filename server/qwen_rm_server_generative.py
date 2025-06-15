import argparse
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

def get_estimate_reward(index, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(
        **model_inputs,
        max_new_tokens=3,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True 
    )
    # yes token's logits
    logits_yes = output.scores[0][0, 9693].item()
    # no token's logits
    logits_no = output.scores[0][0, 2152].item()

    print('yes logits: ', logits_yes)
    print('no logits: ', logits_no)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output.sequences)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(index, logits_yes / (logits_yes + logits_no), response)

    return logits_yes / (logits_yes + logits_no)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    # print(data)
    return jsonify({'response': get_estimate_reward(data['index'], data['messages'])})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--host', type=str, default='')
    parser.add_argument('--port', type=int, default=8081)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    app.run(host=args.host, port=args.port)