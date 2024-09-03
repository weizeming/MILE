import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num', type=int, required=True)
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
    
    l = args.num
    def get_MSS_score(df, dataset_ids, mutator_ids):
        dataset = df.loc[list(dataset_ids), :]
        killed = dataset[mutator_ids].sum(axis=0) > 0
        MSS = killed
        return round(MSS * 100, 1).sum()

    def get_MSG_score(df, dataset_ids, mutator_ids):
        dataset = df.loc[list(dataset_ids), :]
        killed = dataset[mutator_ids] > 0
        killed['score'] = killed[mutator_ids].mean(axis=1)
        # print(killed)
        MSG = killed['score']
        return round(MSG * 100, 1).sum()
    
    
    all_MSS = {}
    all_MSG = {}

    for task in tasks:
        MSS_task_scores = [] # non, uni
        MSG_task_scores = [] # non, uni
        for model in models:
            # print(model, task)
            mutator_df = pd.read_csv(f'results/{model}_{task}.csv', dtype=float, index_col='test_id')
            killed_df = pd.read_csv(f'results/{model}_{task}_detail.csv', dtype=int, index_col='test_id')

            mutator_ids = list(str(i) for i in range(0,120))

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
            uni_score = get_MSS_score(killed_df,uni_dataset,mutator_ids)/len(mutator_ids), get_MSG_score(mutator_df,uni_dataset,mutators)/len(uni_dataset)
            
            
            avg_non_score = [0,0]
            for bias in all_labels:
                for _, label in enumerate(all_labels):
                    non_dataset = []
                    if label==bias:
                        N = min(l//(len(all_labels)*2) + l//2, len(label_ids[label]))
                        non_dataset += random.sample(label_ids[label],N)
                    else:
                        N = min(l//(len(all_labels)*2), len(label_ids[label]))
                        non_dataset += random.sample(label_ids[label],N)
                if len(non_dataset) > len(uni_dataset):
                    non_dataset = non_dataset[:len(uni_dataset)]
                non_score = get_MSS_score(killed_df,non_dataset,mutator_ids)/len(mutator_ids), get_MSG_score(mutator_df,non_dataset,mutators)/len(uni_dataset)
                avg_non_score[0] += non_score[0]
                avg_non_score[1] += non_score[1]         

            avg_non_score[0] /= len(all_labels)
            avg_non_score[1] /= len(all_labels)
            
            non_score = avg_non_score.copy()
            MSS_task_scores.append(uni_score[0])
            MSS_task_scores.append(non_score[0])
            MSG_task_scores.append(uni_score[1])
            MSG_task_scores.append(non_score[1])
        all_MSS[task] = MSS_task_scores
        all_MSG[task] = MSG_task_scores
    indexs = [(model, data) for model in models for data in ['uni', 'non']]
    r = lambda x: round(x,1)
    all_MSS = pd.DataFrame(all_MSS, index=indexs)
    all_MSG = pd.DataFrame(all_MSG, index=indexs)
    all_MSS['Avg'] = all_MSS.mean(axis=1)
    all_MSG['Avg'] = all_MSG.mean(axis=1)
    all_MSS = all_MSS.apply(r)
    all_MSG = all_MSG.apply(r)
    MSS_avg_uni = all_MSS.iloc[[0,2,4]].mean(axis=0)
    MSG_avg_uni = all_MSG.iloc[[0,2,4]].mean(axis=0)
    MSS_avg_non = all_MSS.iloc[[1,3,5]].mean(axis=0)
    MSG_avg_non = all_MSG.iloc[[1,3,5]].mean(axis=0)

    for task in all_MSG.columns:
        print(task, end=' & ')
        for i in range(6):
            print(all_MSS[task].iloc[i], end='\% & ')
        print('\\textbf{',end='')
        print(round(MSS_avg_uni[task],1),'\%} & ', round(MSS_avg_non[task],1), sep='', end='\% \\\\\n')
        
    print('\n')
    for task in all_MSG.columns:
        print(task, end=' & ')
        for i in range(6):
            print(all_MSG[task].iloc[i], end='\% & ')
        print('\\textbf{',end='')
        print(round(MSG_avg_uni[task],1),'\%} & ', round(MSG_avg_non[task],1), sep='', end='\% \\\\\n')


    