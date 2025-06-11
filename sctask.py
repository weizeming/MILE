import argparse
from utils import *
from paths import model_paths
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
np.random.seed(20240829)
from pprint import pprint

device = 'cuda:0'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['STS', 'SST2exp', 'TR', 'FDTP'], default='STS')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=20)
    parser.add_argument('--test_example', type=int, default=300)
    parser.add_argument('--mutants', type=int, default=20)
    return parser.parse_args()


args = get_args()


if __name__ == '__main__':
    task = args.task
    left, right = task_value_interval(task)
    demo_df = pd.read_csv(f'./data/{task}/train.csv')
    test_df = pd.read_csv(f'./data/{task}/val.csv')
    ood_df = pd.read_csv("./data/WMT/val.csv")
    gau_sigma = (right - left)/50
    error_acc = error_accept(task)
    
    task_label_col = {
         'STS':2, 'SST2exp': 1, 'TR': 1, 'FDTP': 8,
    }
    label_col = task_label_col[task]
    
    labels = demo_df.iloc[:, label_col].values
    sorted_idx = np.argsort(labels)
    demo_df = demo_df.iloc[sorted_idx]  # 按标签排序

    row_count = demo_df.shape[0]

    ### 让样例取值均匀分布
    label_shot = args.shots
    if task == 'TR':
        label_shot = label_shot-5
    mutants_num = args.mutants
    if task == 'TR':
        mutants_num = mutants_num-5

    inlen = (row_count-1) / label_shot
    interval_len = inlen // 1
    

    vanilla_demos = []

    for rounds in range(label_shot):
        randomlabel = np.random.randint(interval_len*rounds, interval_len*(rounds+1))
        demo = demo_df.iloc[randomlabel,:]
        sorted_demo = sort_demo(task, demo)
        vanilla_demos.append(sorted_demo)

    vanilla_demos = [vanilla_demos[i] for i in np.random.permutation(len(vanilla_demos))]
    vanilla_template = ICL_Template_Reg(task, vanilla_demos)
    vanilla_ICL_prompt = vanilla_template.get_prompt()
    
    all_mutators = [
        "OOD_label",
        "noisy_label",
        "blurred_input",
        "OOD_demo",
        "demo_shuffle",
        "demo_repitition"
    ]
    
    mutator_configs = {
        "OOD_label": [(i,) for i in range(mutants_num)],
        "noisy_label": [(np.random.normal(0, gau_sigma, size=label_shot),) for i in range(mutants_num)],
        "blurred_input": [(i,) for i in range(mutants_num)],
        "OOD_demo": [(i, ood_df.iloc[i, :]) for i in range(mutants_num)],
        "demo_shuffle": [(np.random.permutation(label_shot),) for _ in range(mutants_num)],
        "demo_repitition": [(i, 2) for i in range(mutants_num)]
    }
    
    mutated_ICL_prompts = {}

    for mutator in all_mutators:
        mutated_ICL_prompts[mutator] = []
        for conf in mutator_configs[mutator]:
            mutator_template = ICL_Template_Reg(task, vanilla_demos)
            getattr(mutator_template, mutator)(*conf)
            mutated_ICL_prompts[mutator].append(mutator_template.get_prompt())
    
        # pprint(mutated_ICL_prompts[mutator])
    
    conv_template = load_conversation_template(model_paths[args.model])
    model, tokenizer = load_model(model_paths[args.model])

    def mutatoion_score(test_id, test_prompt):
        vanilla_prompt = vanilla_ICL_prompt + test_prompt
        vanilla_pred = get_response_value(model, tokenizer, vanilla_prompt)  # 获取预测结果（可能包含数字或文本）
        y = test_df.iloc[test_id, label_col]
        if np.abs(vanilla_pred-y) > error_acc:
            return -1, -1, -1
        
        scores = []
        pred_scr = []  

        for mutator in all_mutators:
            cnt = 0
            killed = 0
            for mutated_prompt in mutated_ICL_prompts[mutator]:
                final_prompt = mutated_prompt + test_prompt
                pred = get_response_value(model, tokenizer, final_prompt)
                pred_scr.append(float(pred))
                if(np.abs(float(pred)-float(y)) > error_accept(task)):
                    killed += 1
                cnt += 1
            scores.append(round(killed/cnt,2))
        return scores, pred_scr, y
    
    all_scores = []
    all_pred_scr = []
    real_answer = []
    for test_id in range(args.test_example):
        test_prompt = format_example(test_df, test_id, task)
        score, pred_scrs, y = mutatoion_score(test_id, test_prompt)
        
        if score == -1:
            print(f'case {test_id} output is wrong answer')
        else:
            print(f'case {test_id} output is right answer')
            all_scores.append([test_id, *score])
            all_pred_scr.append([test_id, *pred_scrs])
            real_answer.append([test_id, y])
            
        if (test_id+1) % 10 == 0:
            result_df = pd.DataFrame(all_scores, columns=['test_id', *all_mutators])
            result_detail = pd.DataFrame(all_pred_scr, columns=['test_id', *range(len(all_mutators) * label_shot)])
            result_real_answer = pd.DataFrame(real_answer, columns=['test_id', 'real_answer'])
            result_df.to_csv(f'./results/{args.model}_{args.task}.csv')
            result_detail.to_csv(f'./results/{args.model}_{args.task}_detail.csv')
            result_real_answer.to_csv(f'./results/{args.model}_{args.task}_realanswer.csv')
            print(f'{len(all_scores)} cases saved. Current test id: {test_id}')
    