import argparse
from utils import *
from paths import model_paths
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
np.random.seed(20240829)

device = 'cuda:0'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['STS', 'SST2exp', 'TR', 'FDTP', 'all'], default='STS')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=20)
    parser.add_argument('--test-example', default=300, type=int)
    return parser.parse_args()

args = get_args()

if __name__ == '__main__':
    task = args.task
    if args.task == 'all':
        rvs = []
        all_tasks = ['STS', 'SST2exp', 'TR', 'FDTP']
        model, tokenizer = load_model(model_paths[args.model])        
        
        for task in all_tasks:
            np.random.seed(20240829)
            demo_df = pd.read_csv(f'./data/{task}/train.csv')
            test_df = pd.read_csv(f'./data/{task}/val.csv')

            task_label_col = {
                'STS':2, 'SST2exp': 1, 'TR': 1, 'FDTP': 8,
            }
            
            left, right = task_value_interval(task)
            label_col = task_label_col[task]
            
            labels = demo_df.iloc[:, label_col].values
            sorted_idx = np.argsort(labels)
            demo_df = demo_df.iloc[sorted_idx]  # 按标签排序
            row_count = demo_df.shape[0]
            
            ### 让样例取值均匀分布
            label_shot = args.shots
            if task == 'TR':
                label_shot = label_shot-5
            interval_len = (row_count-1) // label_shot
            
            vanilla_demos = [demo_df.iloc[i*interval_len] for i in np.random.permutation(label_shot)]
            vanilla_template = ICL_Template_Reg(task, vanilla_demos)
            vanilla_ICL_prompt = vanilla_template.get_prompt()

            def eval_rv(test_ICL_prompt, max_example):
                preds, targets = [], []
                test_cnt = 0 
                for id in range(min(len(test_df), max_example)):
                    test_prompt = format_example(test_df, id, task)
                    final_prompt = test_ICL_prompt + test_prompt
                    pred_value = get_response_value(model, tokenizer, final_prompt)
                    if id == 0:
                        print(final_prompt)
                    print(f'{id}:',pred_value)
                    test_cnt+=1
                    true_value = test_df.iloc[id, label_col]
                    preds.append(float(pred_value))
                    targets.append(float(true_value))

                
                preds, targets = np.array(preds), np.array(targets)
                return   1- ((np.sum(np.abs(preds - targets))) / (test_cnt*(right-left)))  # 计算相对方差

            vanilla_rv = round(eval_rv(vanilla_ICL_prompt, args.test_example), 3)
            rvs.append(vanilla_rv)
        
        rvs = pd.DataFrame([rvs], columns=all_tasks, index=[args.model])
        rvs.to_csv(f'./results/{args.model}_regtask_acc.csv')
        print(f'{args.model}  accuracy has been saved')
    else:
        raise NotImplementedError
