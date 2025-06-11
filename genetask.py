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
    parser.add_argument('--task', choices=['sciq', 'squad'], default='sciq')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=12)
    parser.add_argument('--test_example', type=int, default=300)
    parser.add_argument('--mutants', type=int, default=12)
    return parser.parse_args()


args = get_args()


if __name__ == '__main__':
    task = args.task
    demo_df = pd.read_csv(f'./data/{task}/train.csv')
    test_df = pd.read_csv(f'./data/{task}/val.csv')
    ood_df = pd.read_csv("./data/WMT/val.csv")

    task_label_col = {
         'sciq':1, 'squad':2
    }
    label_col = task_label_col[task]
    
    row_count = demo_df.shape[0]

    ### 让样例取值均匀分布
    label_shot = args.shots
    mutants_num = args.mutants

    inlen = (row_count-1) / label_shot
    interval_len = inlen // 1
    

    vanilla_demos = []

    for rounds in range(label_shot):
        randomlabel = np.random.randint(interval_len*rounds, interval_len*(rounds+1))
        demo = demo_df.iloc[randomlabel,:]
        vanilla_demos.append(demo)

    vanilla_demos = [vanilla_demos[i] for i in np.random.permutation(len(vanilla_demos))]
    vanilla_template = ICL_Template_Gene(task, vanilla_demos)
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
        "noisy_label": [(i,) for i in range(mutants_num)],
        "blurred_input": [(i,) for i in range(mutants_num)],
        "OOD_demo": [(i, ood_df.iloc[i, :]) for i in range(mutants_num)],
        "demo_shuffle": [(np.random.permutation(label_shot),) for _ in range(mutants_num)],
        "demo_repitition": [(i, 2) for i in range(mutants_num)]
    }
    
    mutated_ICL_prompts = {}

    for mutator in all_mutators:
        mutated_ICL_prompts[mutator] = []
        for conf in mutator_configs[mutator]:
            mutator_template = ICL_Template_Gene(task, vanilla_demos)
            getattr(mutator_template, mutator)(*conf)
            mutated_ICL_prompts[mutator].append(mutator_template.get_prompt())
    
        # pprint(mutated_ICL_prompts[mutator])
    
    conv_template = load_conversation_template(model_paths[args.model])
    model, tokenizer = load_model(model_paths[args.model])

    def mutatoion_score(test_id, test_prompt):
        vanilla_prompt = vanilla_ICL_prompt + test_prompt
        vanilla_pred = get_response(model, tokenizer, vanilla_prompt)  # 获取预测结果（可能包含数字或文本）
        y = test_df.iloc[test_id, label_col]
        if not contains_phrase(y, vanilla_pred):
            return -1, -1
        
        scores = []
        pred_scr = []  

        for mutator in all_mutators:
            cnt = 0
            killed = 0
            for mutated_prompt in mutated_ICL_prompts[mutator]:
                final_prompt = mutated_prompt + test_prompt
                pred = get_response(model, tokenizer, final_prompt)
                if not contains_phrase(y, pred):
                    killed += 1
                    pred_scr.append(1) 
                if contains_phrase(y, pred):
                    pred_scr.append(0)
                cnt += 1
            scores.append(round(killed/cnt,2))
        return scores, pred_scr
    
    all_scores = []
    all_pred_scr = []
    for test_id in range(args.test_example):
        test_prompt = format_example(test_df, test_id, task)
        score, pred_scrs = mutatoion_score(test_id, test_prompt)
        
        if score == -1:
            print(f'case {test_id} output is wrong answer')
        else:
            print(f'case {test_id} output is right answer')
            all_scores.append([test_id, *score])
            all_pred_scr.append([test_id, *pred_scrs])
            
        if (test_id+1) % 10 == 0:
            result_df = pd.DataFrame(all_scores, columns=['test_id', *all_mutators])
            result_detail = pd.DataFrame(all_pred_scr, columns=['test_id', *range(len(all_mutators) * label_shot)])
            result_df.to_csv(f'./results/{args.model}_{args.task}.csv')
            result_detail.to_csv(f'./results/{args.model}_{args.task}_detail.csv')
            print(f'{len(all_scores)} cases saved. Current test id: {test_id}')
    