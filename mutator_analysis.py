import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import matplotlib

from matplotlib.pyplot import MultipleLocator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    models = ['vicuna', 'llama', 'falcon']
    tasks = ['SST2','AGnews','mrpc','QNLI','RTE']
    if args.task is not None:
        tasks = [args.task]

    mutators = ['noisy_label','OOD_label','blurred_input','demo_shuffle','OOD_demo','demo_repitition']
    
    test_dfs = {
        task: pd.read_csv(f'data/{task}/test.csv') for task in tasks
    }
    
    task_label_col = {
        'SST2':1, 'AGnews':0, 'RTE':2, 'mrpc':2, 'QNLI':2,
    }
    
    
    def get_MSS_score(df, dataset_ids, mutator_ids):
        dataset = df.loc[list(dataset_ids), :]
        killed = dataset[mutator_ids].sum(axis=0) > 0
        MSS = killed
        return round(MSS * 100, 1).sum()

    def get_MSG_analysis(df, dataset_ids, mutator_ids):
        dataset = df.loc[list(dataset_ids), :]
        killed = dataset[mutator_ids] > 0
        return killed.mean(0)
        

    l = 50

    all_MSG = {}
    
    for task in tasks:
        MSG_task_scores = {}
        for model in models:

            mutator_df = pd.read_csv(f'results/{model}_{task}.csv', dtype=float, index_col='test_id')
            killed_df = pd.read_csv(f'results/{model}_{task}_detail.csv', dtype=int, index_col='test_id')

            test_ids = mutator_df.index
            label_col = task_label_col[task]
            labels = test_dfs[task].iloc[test_ids, label_col]
            all_labels = labels.unique()
            
            label_ids = {}

            for label in all_labels:
                label_id = labels==label
                label_id = [i for i in label_id.index if label_id[i]]
                label_ids[label] = label_id
                continue

            uni_dataset = []
            for label in all_labels:
                N = min(l//len(all_labels), len(label_ids[label]))
                uni_dataset += random.sample(label_ids[label],N)
            
            MSG_analysis = get_MSG_analysis(mutator_df, uni_dataset, mutators)
            MSG_task_scores[model] = MSG_analysis
        MSG_task_scores = pd.DataFrame(MSG_task_scores)
        r = lambda x: round(100*x,1)
        MSG_task_scores = MSG_task_scores.apply(r)
        # print(MSG_task_scores)
        all_MSG[task] = MSG_task_scores
        
    all_MSG['Avg'] = all_MSG['SST2'] * 0
    for task in tasks:
        all_MSG['Avg'] += all_MSG[task]
    all_MSG['Avg'] /= len(tasks)
    
    all_MSG['Avg']['Avg'] = all_MSG['Avg'].mean(axis=1)
    r = lambda x: round(x,1)
    all_MSG['Avg'] = all_MSG['Avg'].apply(r)
    
    print(all_MSG)
    

    matplotlib.rcParams["font.size"] = 17
    y_major_locator=MultipleLocator(10)

    tasks.append('Avg')
    for task in tasks:
        data = all_MSG[task].mean(axis=1)
        y = list(data)
        x = ['NL', 'OL', 'BI', 'DS', 'OD', 'DR']
        blue_palette = ['r','g','b','c','m','y']
        
        
        plt.bar(x, y, color=blue_palette)
        plt.ylim(0,90)
        plt.title(task, fontsize=19)
        plt.savefig(f'figs/mutator_{task}.png', dpi=200, bbox_inches='tight')
        plt.clf()
        
        
