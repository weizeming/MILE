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
    parser.add_argument('--task', choices=['sciq', 'squad', 'all'], default='sciq')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=15)
    parser.add_argument('--test-example', default=300, type=int)
    return parser.parse_args()

args = get_args()

if __name__ == '__main__':
    task = args.task
    if args.task == 'all':
        rvs = []
        all_tasks = ['sciq', 'squad']
        model, tokenizer = load_model(model_paths[args.model])        
        
        for task in all_tasks:
            np.random.seed(20240829)
            demo_df = pd.read_csv(f'./data/{task}/train.csv')
            test_df = pd.read_csv(f'./data/{task}/val.csv')

            task_label_col = {
                'sciq':1, 'squad':2
            }
            
            label_col = task_label_col[task]
            
            row_count = demo_df.shape[0]
            
            ### 让样例取值均匀分布
            label_shot = args.shots
            interval_len = (row_count-1) // label_shot
            
            vanilla_demos = [demo_df.iloc[i*interval_len] for i in np.random.permutation(label_shot)]
            vanilla_template = ICL_Template_Gene(task, vanilla_demos)
            vanilla_ICL_prompt = vanilla_template.get_prompt()
            
            def eval_rv(test_ICL_prompt, max_example):
                preds = 0
                test_cnt = 0 
                for id in range(min(len(test_df), max_example)):
                    test_prompt = format_example(test_df, id, task)
                    final_prompt = test_ICL_prompt + test_prompt
                    pred_value = get_response(model, tokenizer, final_prompt)
                    if id == 0:
                        print(final_prompt)
                    print(f'{id}:',pred_value)
                    print(pred_value)
                    test_cnt+=1   
                    true_value = test_df.iloc[id, label_col]
                    if contains_phrase(true_value, pred_value):
                        preds += 1
                    print(contains_phrase(true_value, pred_value))
                return   preds/test_cnt 
            vanilla_rv = round(eval_rv(vanilla_ICL_prompt, args.test_example), 3)
            rvs.append(vanilla_rv)
        
        rvs = pd.DataFrame([rvs], columns=all_tasks, index=[args.model])
        rvs.to_csv(f'./results/{args.model}_genetask_acc.csv')
        print(f'{args.model}  accuracy has been saved')
    else:
        raise NotImplementedError
