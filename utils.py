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
import re

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = 'cuda:0'
def contains_phrase(str_list, target_str):
    phrases = [phrase.strip().lower() for phrase in str_list.split(",")]  # 分割并去除空格
    target_str = target_str.lower()  # 统一转换为小写
    return any(phrase in target_str for phrase in phrases)

def task_value_interval(task):
    if task == 'STS' :
        return float(0), float(1)

    if task == 'SST2exp':
        return float(0), float(4)
    
    if task == 'TR':
        return float(1), float(5)
    
    if task == 'FDTP':
        return float(10), float(54)

def error_accept(task):
    if task == 'STS' :
        return float(0.25)

    if task == 'SST2exp':
        return float(1)
    
    if task == 'TR':
        return float(1)
    
    if task == 'FDTP':
        return float(11)
    
def interval_len(task):
    if task == 'STS' :
        return float(1)

    if task == 'SST2exp':
        return float(4)
    
    if task == 'TR':
        return float(4)
    
    if task == 'FDTP':
        return float(44)
    
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

### 对回归任务的变异
class ICL_Template_Reg:
    def __init__(self, task, demos) -> None:
        self.task = task
        self.demos = [list(d).copy() for d in demos]
    
    def noisy_label(self, noise):
        """
        对回归任务的标签添加高斯噪声
        """
        left, right = task_value_interval(self.task)
        for i in range(len(self.demos)):
            if self.demos[i][-1]+noise[i]>left and self.demos[i][-1]+noise[i]<right:
                 self.demos[i][-1] = self.demos[i][-1]+noise[i]
            self.demos[i][-1] = np.round(self.demos[i][-1], 3)
    
    def OOD_label(self, demo_id):
        """
        将回归标签替换为不合理的数值
        """
        left, right = task_value_interval(self.task)
        self.demos[demo_id][-1] = right*10
    
    def blurred_input(self, demo_id):
        """
        模糊输入，例如截断文本或删除部分内容
        """
        if self.task == 'FDTP':
            for input_pos in range(len(self.demos[demo_id]) - 1):
                if input_pos >= 6:
                    l = len(self.demos[demo_id][input_pos])
                    self.demos[demo_id][input_pos] = self.demos[demo_id][input_pos][: l // 2]
        else :
            for input_pos in range(len(self.demos[demo_id]) - 1):
                l = len(self.demos[demo_id][input_pos])
                self.demos[demo_id][input_pos] = self.demos[demo_id][input_pos][: l // 2]
                
    def demo_shuffle(self, new_order):
        """
        变换示例顺序，检查模型对顺序变化的敏感度
        """
        temp = [self.demos[i] for i in new_order]
        self.demos = temp.copy()
    
    def OOD_demo(self, demo_id, new_demo):
        """
        在提示中插入来自不同任务的数据点
        """
        for i in range(len(self.demos[demo_id]) - 1):
            j = i%2
            self.demos[demo_id][i] = new_demo[j]
    
    def demo_repitition(self, demo_id, num=2):
        """
        重复某些示例，检查模型是否受重复示例影响
        """
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

class ICL_Template_Gene:
    def __init__(self, task, demos) -> None:
        self.task = task
        self.demos = [list(d).copy() for d in demos]
        
    def OOD_label(self, demo_id):
        if(self.task == 'sciq'):
            self.demos[demo_id][1] = '#'
        else:
            self.demos[demo_id][-1] = '#'

    def noisy_label(self, demo_id):
        if(self.task == 'sciq'):
            self.demos[demo_id][1] = self.demos[demo_id][1] + ' ' + "###"
        else:
            self.demos[demo_id][-1] = self.demos[demo_id][-1] + ' ' + "###"
         
    def blurred_input(self, demo_id):
        if(self.task == 'sciq'):
            self.demos[demo_id][0] = str(self.demos[demo_id][0])
            l = len(self.demos[demo_id][0])
            self.demos[demo_id][0] = self.demos[demo_id][0][:l // 2]
            self.demos[demo_id][2] = str(self.demos[demo_id][2])
            l = len(self.demos[demo_id][2])
            self.demos[demo_id][2] = self.demos[demo_id][2][:l // 2]
        else:
            for input_pos in range(len(self.demos[demo_id]) - 1):
                self.demos[demo_id][input_pos] = str(self.demos[demo_id][input_pos])
                l = len(self.demos[demo_id][input_pos])
                self.demos[demo_id][input_pos] = self.demos[demo_id][input_pos][:l // 2]

    def OOD_demo(self, demo_id, new_demo):
        if(self.task == 'sciq'):
            self.demos[demo_id][0] = new_demo[0]
            self.demos[demo_id][2] = new_demo[1]
        else:
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
            right_demo = sort_demo(self.task, demo)
            for id, token in enumerate(template):
                prompt += token
                prompt += str(right_demo[id])
            prompt += '\n\n' 
        return prompt
        
def regression_dataset_preprocess(task, val_num):
    df = pd.read_csv(f'./data/{task}/{task}.csv')
    if task == 'SST2exp':
        df = df.drop(['label_text'], axis=1)
    if task == 'FDTP':
        df = df.drop(['ID', 'Delivery_person_ID'], axis=1)

    target_index ={'STS':'score', 'SST2exp':'label', 'TR':'Rating', 'FDTP':'Time_taken(min)'}
    df = df.sort_values(by=target_index[task]).reset_index(drop=True)

    total_val_count = len(df)
    step = (total_val_count-1) // val_num  # 计算步长
    val_idx = list(range(0, len(df), step))  
    val_df = df.iloc[val_idx]  # 验证集
    train_df = df.drop(index=val_idx)  # 训练集
    
    train_df.to_csv(f'./data/{task}/train.csv', index=False)
    val_df.to_csv(f'./data/{task}/val.csv', index=False)

    print(f'The training and valuation datasets for {task} have been ready')

def generation_dataset_preprocess(task, val_num, dataset):
    if dataset == 'train':
        df = pd.read_csv(f'./data/{task}/{task}_train_org.csv')
    if dataset == 'val':
        df = pd.read_csv(f'./data/{task}/{task}_val_org.csv')
    
    if task == 'sciq':
        df = df.drop(['distractor1', 'distractor2', 'distractor3'], axis=1)
    if task == 'squad':
        df = df.drop(['id', 'title'], axis=1) 
        df["answers"] = df["answers"].apply(lambda x: re.search(r"\{'text': \[(.*?)\]", x).group(1) if re.search(r"\{'text': \[(.*?)\]", x) else x)
    if dataset == 'train':
        df.to_csv(f'./data/{task}/train.csv', index=False) 
        print(f'The training datasets for {task} have been ready')
    if dataset == 'val':
        total_val_count = len(df)
        step = (total_val_count-1) // val_num  # 计算步长
        val_idx = list(range(0, len(df), step))  
        val_df = df.iloc[val_idx]  # 验证集
        val_df.to_csv(f'./data/{task}/val.csv', index=False)
        print(f'The valuation datasets for {task} have been ready')
    
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
    ### 回归任务的task
    if task == 'STS':
        prompt = 'Please compare the two sentences and rate their similarity on a scale from 0 to 1,just output only one float value.\n\n'
        return prompt
    if task == 'SST2exp':
        prompt = 'Please rate the emotional intensity of the text on a scale from 0 to 4,just output only one float value.\n\n'
        return prompt
    if task == 'TR':
        prompt = 'Please predict a traveler rating of travel-related services based on their reviews of hotels, restaurants, and tourist attractions, on a scale from 1 to 5,just output only one float value.\n\n'
        return prompt
    if task == 'FDTP':
        prompt = 'Please predict the delivery time based on delivery age, delivery person ratings, restaurant latitude, restaurant location, delivery location, type of order, type of vehicle, on a scale from 10 to 54,just output only one float value.\n\n'
        return prompt
    ### 生成任务的prompt
    if task == 'sciq':
        prompt = 'Please answer the question based on the support, just output only the answer.\n\n'
        return prompt
    if task == 'squad':
        prompt = 'Please answer the question based on the context, just output only the answer.\n\n'
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
    ### 回归任务数据集
    if task == 'STS' :
        return demo_df[0], demo_df[1], demo_df[2]
    
    if task == 'SST2exp':
        return demo_df[0], demo_df[1]
    
    if task == 'TR':
        return demo_df[0], demo_df[1]
    
    if task == 'FDTP':
        return demo_df[0], demo_df[1], demo_df[2], demo_df[3], demo_df[4], demo_df[5], demo_df[6], demo_df[7], demo_df[8]
    ### 生成任务
    if task == 'sciq':
        return demo_df[2], demo_df[0], demo_df[1]
    
    if task == 'squad':
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
    ### 回归任务数据集
    if task == 'STS' :
        return df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2]

    if task == 'SST2exp':
        return df.iloc[idx, 0], df.iloc[idx, 1]
    
    if task == 'TR':
        return df.iloc[idx, 0], df.iloc[idx, 1]
    
    if task == 'FDTP':
        return df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4], df.iloc[idx, 5], df.iloc[idx, 6], df.iloc[idx, 7], df.iloc[idx, 8]
    if task == 'sciq':
        return df.iloc[idx, 2], df.iloc[idx, 0], df.iloc[idx, 1]
    
    if task == 'squad':
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
    ### 回归任务数据集
     
    if task == 'STS' :
        return "Sentence 1:", "\nSentence 2:", "\nSimilarity Score:"

    if task == 'SST2exp':
        return "Text:", "\nEmotional Intensity:"
    
    if task == 'TR':
        return "Review:", "\nRating:"
    
    if task == 'FDTP':
        return "Delivery person Age:", "\nDelivery person Ratings:", "\nRestaurant latitude:", "\nRestaurant longitude:", "\nDelivery location latitude:", "\nDelivery location longitude:", "\nType of order:", "\nType of vehicle:", "\nTime taken:"
    ###生成任务
    if task == 'sciq':
        return "Support:", "\nQuestion:", "\nAnswer:"
    
    if task == 'squad':
        return "Context:", "\nQuestion:", "\nAnswer:"
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
    ### 回归任务数据集
     
    if task == 'STS' :
        prompt = "Sentence 1:"+str(df.iloc[idx, 0])+"\nSentence 2:"+str(df.iloc[idx, 1])
        prompt += "\nSimilarity Score:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 2]))
        return prompt

    if task == 'SST2exp':
        prompt = "Text:"+str(df.iloc[idx, 0])
        prompt += "\nEmotional Intensity:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 1]))
        return prompt
    
    if task == 'TR':
        prompt = "Review:"+str(df.iloc[idx, 0])
        prompt += "\nRating:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 1]))
        return prompt
    
    if task == 'FDTP':
        prompt = "Delivery person Age:"+str(df.iloc[idx, 0])+"\nDelivery person Ratings:"+str(df.iloc[idx, 1])+ "\nRestaurant latitude:"+str(df.iloc[idx, 2])+"\nRestaurant longitude:"+str(df.iloc[idx, 3])+"\nDelivery location latitude:"+str(df.iloc[idx, 4])+"\nDelivery location longitude:"+str(df.iloc[idx, 5])+"\nType of order:"+str(df.iloc[idx, 6])+"\nType of vehicle:"+str(df.iloc[idx, 7])
        prompt += "\nTime taken:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 8]))
        return prompt
    ###生成任务：
    if task == 'sciq':
        prompt = "Support:"+str(df.iloc[idx, 2])+"\nQuestion:"+str(df.iloc[idx, 0])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 1]))
        return prompt
    if task == 'squad':
        prompt = "Context:"+str(df.iloc[idx, 0])+"\nQuestion:"+str(df.iloc[idx, 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(str(df.iloc[idx, 2]))
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

def get_response_value(model, tokenizer, prompt):
    response_value = None
    vanilla_pred = get_response(model, tokenizer, prompt)
    def value_match(prompt):
        pattern = r'-?\d+(\.\d+)?([eE][-+]?\d+)?'
        match = re.search(pattern, prompt)
        if match:
            first_number = match.group()  # 获取匹配的第一个数值
            return float(first_number)
        else:
            return None
        
    for part in vanilla_pred.split():  # 分割字符串
        response_value = value_match(part)
        if response_value is not None:
            break
    return round(response_value, 2)

def generate(model, tokenizer, input_ids, assistant_role_slice=None, gen_config=None, max_tokens=None, debug=False):
    
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 5
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