import argparse
from utils import *
from paths import model_paths
import torch
device = 'cuda:0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vicuna')
    args = parser.parse_args()
    conv_template = load_conversation_template(model_paths[args.model])
    model, tokenizer = load_model(model_paths[args.model])
    print("Model is ready.")
    while True:
        user_prompt = input()
        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], '')
        prompt = conv_template.get_prompt()
        # print(prompt)
        input_ids = tokenizer(prompt).input_ids
        input_ids = torch.tensor(input_ids).to(device)
        output_ids = generate(model, tokenizer, input_ids)[0]
        # print(output_ids, input_ids)
        output_ids = output_ids[len(input_ids):]
        generate_str = tokenizer.decode(output_ids).strip()
        conv_template.update_last_message(generate_str)

        print(generate_str)