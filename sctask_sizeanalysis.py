import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['STS', 'SST2exp', 'TR', 'FDTP'], type=str, default=None)
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], type=str, default=None)
    parser.add_argument('--seed', type=int, default=44)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    models = ['vicuna', 'llama', 'falcon']
    tasks = ['STS', 'SST2exp', 'TR', 'FDTP']
    if args.task is not None:
        tasks = [args.task]

    mutators = ['noisy_label','OOD_label','blurred_input','demo_shuffle','OOD_demo','demo_repitition']
    
    test_dfs = {
        task: pd.read_csv(f'data/{task}/val.csv') for task in tasks
    }
    
    task_label_col = {
        'STS':2, 'SST2exp': 1, 'TR': 1, 'FDTP': 8,
    }

    mutator_num = {
        'STS':20, 'SST2exp': 20, 'TR': 15, 'FDTP': 20,
    }
    
    def constrained_sample(population, N, must_include):
        must_include = set(must_include)  # 确保 must_include 是集合
        remaining = list(set(population) - must_include)  # 剩余可采样的元素

        # 如果 must_include 的数量超过 N，直接截断
        if len(must_include) > N:
            return list(must_include)[:N]
        
        # 采样剩余的元素
        extra_samples = random.sample(remaining, k=min(N - len(must_include), len(remaining)))

        return list(must_include) + extra_samples
    
    def get_MSSR_score(df, dataset_ids, mutator_ids, task):
        MSSR = 0
        for mutator in mutator_ids:
            aaa = []
            
            for id in dataset_ids:
                aaa.append(np.abs(df.iloc[int(id)].loc[mutator]-test_dfs[task].iloc[int(id)].iloc[int(task_label_col[task])]) / interval_len(task))
                
            MSSR += max(aaa)
        MSSR = MSSR/(len(mutator_ids))
        return round(MSSR * 100, 2)

    def get_MSGR_score(df, dataset_ids, mutators_name, task):
        MSGR = 0
        for id in dataset_ids:
            for i in range(6):
                aaa = []
                for j in [str(t) for t in range(i*mutator_num[task], (i+1)*mutator_num[task])]:
                    aaa.append(np.abs(df.iloc[int(id)].loc[j]-test_dfs[task].iloc[int(id)].iloc[int(task_label_col[task])]) / interval_len(task))
                MSGR += max(aaa)
        MSGR = MSGR/(len(dataset_ids)*len(mutators_name))
        return round(MSGR * 100, 2)
    
    uni_id = {task:{model:{y:[] for y in ['interval_a', 'interval_b', 'interval_c', 'interval_d']} for model in models} for task in tasks}
    nonuni_id = {task:{model:{r:{y:[] for y in ['interval_a', 'interval_b', 'interval_c', 'interval_d']} for r in ['interval_a', 'interval_b', 'interval_c', 'interval_d']}for model in models} for task in tasks}
    
    for l in [20,40,60,80,100,120]:
        all_MSS = {}
        all_MSG = {}

        for task in tasks:
            MSS_task_scores = [] # non, uni
            MSG_task_scores = [] # non, uni
            for model in models:
                # print(model, task)
                mutator_df = pd.read_csv(f'results/{model}_{task}.csv')
                killed_df = pd.read_csv(f'results/{model}_{task}_detail.csv')
                realanswer_df = pd.read_csv(f'results/{model}_{task}_realanswer.csv')
                mutator_ids = list(str(i) for i in range(0,6*mutator_num[task]))

                intlen = interval_len(task)/4
                left, right = task_value_interval(task)
                all_labels = ['interval_a', 'interval_b', 'interval_c', 'interval_d']
                label_ids = {'interval_a':[], 'interval_b':[], 'interval_c':[], 'interval_d':[]}
                test_id_list = mutator_df['test_id'].to_numpy()

                for possible_id in test_id_list:
                    ex_answer = realanswer_df.loc[realanswer_df["test_id"] == possible_id, "real_answer"].iloc[0]
                    if ex_answer >= 0*intlen+left and ex_answer < 1*intlen+left:
                        label_ids['interval_a'].append(possible_id)
                    if ex_answer >= 1*intlen+left and ex_answer < 2*intlen+left:
                        label_ids['interval_b'].append(possible_id)
                    if ex_answer >= 2*intlen+left and ex_answer < 3*intlen+left:
                        label_ids['interval_c'].append(possible_id)
                    if ex_answer >= 3*intlen+left and ex_answer <= 4*intlen+left:
                        label_ids['interval_d'].append(possible_id)

                uni_dataset_id = []
                for label in all_labels:
                    N = min(l//len(all_labels), len(label_ids[label]))
                    t = constrained_sample(label_ids[label],N,uni_id[task][model][label])
                    uni_dataset_id += t
                    uni_id[task][model][label] = t
                uni_dataset = realanswer_df.index[realanswer_df["test_id"].isin(uni_dataset_id)].tolist()
                uni_score = get_MSSR_score(killed_df,uni_dataset,mutator_ids, task), get_MSGR_score(killed_df,uni_dataset,mutators, task)
                
                
                avg_non_score = [0,0]
                for bias in all_labels:
                    non_dataset_id = []
                    for _, label in enumerate(all_labels):
                        if label==bias:
                            N = min(l//(len(all_labels)*4) + (3*l)//4, len(label_ids[label]))
                            t = constrained_sample(label_ids[label],N,nonuni_id[task][model][bias][label])
                            non_dataset_id += t
                            nonuni_id[task][model][bias][label] = t
                        else:
                            N = min(l//(len(all_labels)*4), len(label_ids[label]))
                            t = constrained_sample(label_ids[label],N,nonuni_id[task][model][bias][label])
                            non_dataset_id += t
                            nonuni_id[task][model][bias][label] = t
                    if len(non_dataset_id) > len(uni_dataset_id):
                        non_dataset_id = non_dataset_id[:len(uni_dataset_id)]
                    non_dataset = realanswer_df.index[realanswer_df["test_id"].isin(non_dataset_id)].tolist()
                    non_score = get_MSSR_score(killed_df,non_dataset,mutator_ids, task), get_MSGR_score(killed_df,non_dataset,mutators, task)
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


    