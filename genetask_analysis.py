import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['sciq', 'squad'], type=str, default=None)
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], type=str, default=None)
    parser.add_argument('--seed', type=int, default=44)
    return parser.parse_args()


if __name__ == '__main__': 
    args = get_args()
    random.seed(args.seed)
    models = ['vicuna', 'llama', 'falcon']
    tasks = ['sciq', 'squad']
    if args.task is not None:
        tasks = [args.task]

    mutators = ['noisy_label','OOD_label','blurred_input','demo_shuffle','OOD_demo','demo_repitition']
    
    test_dfs = {
        task: pd.read_csv(f'data/{task}/val.csv') for task in tasks
    }
    
    task_label_col = {
        'sciq':1, 'squad':2  
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
                aaa.append(df.iloc[int(id)].loc[mutator])
            MSSR += max(aaa)
        MSSR = MSSR/(len(mutator_ids))
        return round(MSSR * 100, 2)

    def get_MSGR_score(df, dataset_ids, mutators_name):
        MSGR = 0
        for mutator in mutators_name:
            for id in dataset_ids:
                if df.iloc[int(id)].loc[mutator] > 0:
                    MSGR += 1
        MSGR = MSGR/(len(dataset_ids)*len(mutators_name))
        return round(MSGR * 100, 2)
    uni_id = {task:{model:[] for model in models} for task in tasks}
   
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
                mutator_ids = list(str(i) for i in range(0,72))
                test_id_list = mutator_df['test_id'].to_numpy()
                id_list = []
                for id in test_id_list:
                    id_list.append(id)

                uni_dataset_id = []
                N = min(l, len(id_list))
                t = constrained_sample(id_list,N,uni_id[task][model])
                uni_dataset_id += t
                uni_id[task][model] = t
                uni_dataset = mutator_df.index[mutator_df["test_id"].isin(uni_dataset_id)].tolist()
                uni_score = get_MSSR_score(killed_df,uni_dataset,mutator_ids, task), get_MSGR_score(mutator_df,uni_dataset,mutators)
                print(f'model:{model} task:{task} dataset_num:{l} MSSR:{uni_score[0]} MSGR:{uni_score[1]}')