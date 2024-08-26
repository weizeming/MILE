import torch
import argparse
import os
import json
import pandas as pd
import logging
import fastchat
from fastchat import conversation
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = 'cuda:0'

class ICL_Template:
    def __init__(self, task, demos) -> None:
        self.task = task
        self.demos = [list(d).copy() for d in demos]
        
    def OOD_label(self, demo_id):
        self.demos[demo_id][-1] = '#'

    def noisy_label(self, demo_id, all_labels):
        current_label = self.demos[demo_id][-1]
        current_index = np.where(all_labels == current_label)
        shift_index = (current_index[0] + 1) % len(all_labels)  
        shift_label = all_labels[shift_index]
        self.demos[demo_id][-1] = shift_label[0]

    def blurred_input(self, demo_id):
        for input_pos in range(len(self.demos[demo_id]) - 1):
            l = len(self.demos[demo_id][input_pos])
            self.demos[demo_id][input_pos] = self.demos[demo_id][input_pos][:l // 2]

    def OOD_demo(self, demo_id, new_demo):
        for i in range(len(self.demos[demo_id]) - 1):
            self.demos[demo_id][i] = new_demo[i]

    def demo_shuffle(self, new_order):
        temp = [self.demos[i] for i in new_order]
        self.demos = temp.copy()

    def demo_repitition(self, demo_id, num=2):
        for _ in range(num):
            self.demos.insert(demo_id, self.demos[demo_id])

    def get_prompt(self):
        prompt = get_system_prompt(self.task)
        template = get_template(self.task)
        for demo in self.demos:
            assert len(demo) == len(template)
            for id, token in enumerate(template):
                prompt += token
                prompt += str(demo[id])
            prompt += '\n\n' 
        return prompt

def get_system_prompt(task):
    if task == 'SST2':
        prompt = "The following are multiple film reviews with answers(← or →).\n\n"
        return prompt
    if task == 'AGnews':
        prompt = "Classify the news articles into the categories of 1, 2, 3, or 4.\n\n"
        return prompt
    if task == 'RTE':
        prompt = "Determine whether the hypotheses made based on the premises below are ↑ or ↓.\n\n"
        return prompt
    if task == 'mrpc':
        prompt = "Assess if each pair reflects a semantic equivalence relationship. Use ← or → to indicate the answers.\n\n"
        return prompt
    if task == 'QNLI':
        prompt = "Please determine whether the paragraph contains the answer to the corresponding question. Use ↑ or ↓ to indicate the answers.\n\n"
        return prompt

def sort_demo(task, demo_df):
    if task == 'SST2':
        return demo_df[0], demo_df[1]

    if task == 'AGnews':
        return demo_df[1], demo_df[2], demo_df[0]

    if task == 'RTE':
        return demo_df[0], demo_df[1], demo_df[2]

    if task == 'mrpc':
        return demo_df[0], demo_df[1], demo_df[2]

    if task == 'QNLI':
        return demo_df[0], demo_df[1], demo_df[2]

def get_source(df, idx, task):
    if task == 'SST2':
        return df.iloc[idx, 0], df.iloc[idx, 1]

    if task == 'AGnews':
        return df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 0]

    if task == 'RTE':
        return df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2]

    if task == 'mrpc':
        return df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2]

    if task == 'QNLI':
        return df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2]

def get_template(task):
    if task == 'SST2':
        return "Review:", "\nAnswer:"

    if task == 'AGnews':
        return "Title:", "\nDescription:", "\nAnswer:"

    if task == 'RTE':
        return "Premise:", "\nHypothesis:", "\nAnswer:"

    if task == 'mrpc':
        return "Sentence 1:", "\nSentence 2:", "\nAnswer:"

    if task == 'QNLI':
        return "Question:", "\nParagraph:", "\nAnswer:"

def format_example(df, idx, task, include_answer=False):
    if task == 'SST2':
        prompt = "Review:"+df.iloc[idx, 0]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 1])
        return prompt
    if task == 'AGnews':
        prompt = "Title:"+df.iloc[idx, 1]+"\nDescription:"+df.iloc[idx, 2]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 0])
        return prompt
    if task == 'RTE':
        prompt = "Premise:"+df.iloc[idx, 0]+"\nHypothesis:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt
    if task == 'mrpc':
        prompt = "Sentence 1:"+df.iloc[idx, 0]+"\nSentence 2:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt
    if task == 'QNLI':
        prompt = "Question:"+df.iloc[idx, 0]+"\nParagraph:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt


def get_response(model, tokenizer, prompt):
    # prompt = conv_template.get_prompt()
    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.tensor(input_ids).to(device)
    output_ids = generate(model, tokenizer, input_ids)[0]
    # print(output_ids, input_ids)
    output_ids = output_ids[len(input_ids):]
    generate_str = tokenizer.decode(output_ids).strip()
    # conv_template.update_last_message(generate_str)
    return generate_str

def generate(model, tokenizer, input_ids, assistant_role_slice=None, gen_config=None, max_tokens=None, debug=False):
    
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 4
    if max_tokens is not None:
        gen_config.max_new_tokens = max_tokens
    gen_config.temperature = 0.1

    
    if assistant_role_slice is not None:
        input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    else:
        input_ids = input_ids.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id
                                )
    if assistant_role_slice is not None and not debug:    
        return output_ids[assistant_role_slice.stop:]
    else:
        return output_ids

def load_conversation_template(model_path):
    if 'Llama' in model_path:
        template_name = 'llama-2'
    elif 'vicuna' in model_path:
        template_name = 'vicuna_v1.1'
    elif 'falcon' in model_path:
        template_name = 'falcon-chat'
    elif 'qwen' in model_path:
        template_name = 'qwen-7b-chat'
    else:
        raise NotImplementedError
    conv_template = conversation.get_conv_template(template_name)
    # print(conv_template)
    return conv_template


def load_model(model_path):
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="sequential",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device).eval()

    tokenizer_path = model_path 

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    return model, tokenizer