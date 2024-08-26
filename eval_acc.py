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
    parser.add_argument('--task', choices=['AGnews','mrpc','QNLI','RTE','SST2','all'], default='SST2')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=20)
    parser.add_argument('--test-example', default=500, type=int)
    return parser.parse_args()

args = get_args()

if __name__ == '__main__':
    task = args.task
    if args.task == 'all':
        accs = []
        all_tasks = ['SST2', 'AGnews', 'RTE', 'mrpc', 'QNLI']
        # conv_template = load_conversation_template(model_paths[args.model])
        model, tokenizer = load_model(model_paths[args.model])        
        for task in all_tasks:
            np.random.seed(20240829)
            demo_df = pd.read_csv(f'./data/{task}/val.csv')
            test_df = pd.read_csv(f'./data/{task}/test.csv')
            
            task_label_col = {
                'SST2':1, 'AGnews':0, 'RTE':2, 'mrpc':2, 'QNLI':2,
            }
            
            label_col = task_label_col[task]
            
            labels = demo_df.iloc[:, label_col]
            all_labels = labels.unique()
            assert args.shots % len(all_labels) == 0
            label_shot = args.shots // len(all_labels)

            demo_by_label = [
                demo_df[demo_df.iloc[:, label_col] == all_labels[i]] for i in range(len(all_labels))
            ]

            vanilla_demos = []
            for rounds in range(label_shot):
                for label_id, label in enumerate(all_labels):
                    demo = demo_by_label[label_id].iloc[rounds,:]
                    sorted_demo = sort_demo(task, demo)
                    vanilla_demos.append(sorted_demo)

            vanilla_demos = [vanilla_demos[i] for i in np.random.permutation(len(vanilla_demos))]
            vanilla_template = ICL_Template(task, vanilla_demos)
            vanilla_ICL_prompt = vanilla_template.get_prompt()            

            def eval_acc(test_ICL_prompt, max_example):
                cnt, acc = 0, 0
                for id in range(min(len(test_df), max_example)):
                    test_prompt = format_example(test_df, id, task)
                    final_prompt = test_ICL_prompt + test_prompt
                    pred = get_response(model, tokenizer, final_prompt)
                    y = test_df.iloc[id, label_col]
                    if str(y) in pred:
                        acc += 1
                    cnt += 1
                    # print(y, pred)
                    # print(pred)
                return acc / cnt

            vanilla_acc = eval_acc(vanilla_ICL_prompt, args.test_example)
            print(vanilla_acc)
            accs.append(vanilla_acc)
        
        accs = pd.DataFrame([accs], columns=all_tasks, index=[args.model])
        accs.to_csv(f'./results/{args.model}_acc.csv')

    else:
        raise NotImplementedError