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
    parser.add_argument('--task', choices=['AGnews','mrpc','QNLI','RTE','SST2'], default='SST2')
    parser.add_argument('--model', choices=['vicuna', 'llama', 'falcon'], default='vicuna')
    parser.add_argument('--shots', type=int, default=20)
    parser.add_argument('--test-example', type=int, default=250)
    parser.add_argument('--mutants', type=int, default=20)
    return parser.parse_args()


args = get_args()


if __name__ == '__main__':
    task = args.task
    demo_df = pd.read_csv(f'./data/{task}/val.csv')
    test_df = pd.read_csv(f'./data/{task}/test.csv')
    ood_df = pd.read_csv("./data/WMT/val.csv")
    
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
    
    all_mutators = [
        "OOD_label",
        "noisy_label",
        "blurred_input",
        "OOD_demo",
        "demo_shuffle",
        "demo_repitition"
    ]
    
    mutator_configs = {
        "OOD_label": [(i,) for i in range(args.mutants)],
        "noisy_label": [(i, all_labels) for i in range(args.mutants)],
        "blurred_input": [(i,) for i in range(args.mutants)],
        "OOD_demo": [(i, ood_df.iloc[i, :]) for i in range(args.mutants)],
        "demo_shuffle": [(np.random.permutation(args.shots),) for _ in range(args.mutants)],
        "demo_repitition": [(i, 2) for i in range(args.mutants)]
    }
    
    mutated_ICL_prompts = {}
    for mutator in all_mutators:
        mutated_ICL_prompts[mutator] = []
        for conf in mutator_configs[mutator]:
            mutator_template = ICL_Template(task, vanilla_demos)
            getattr(mutator_template, mutator)(*conf)
            mutated_ICL_prompts[mutator].append(mutator_template.get_prompt())
    
        # pprint(mutated_ICL_prompts[mutator])
    
    conv_template = load_conversation_template(model_paths[args.model])
    model, tokenizer = load_model(model_paths[args.model])

    def mutatoion_score(test_id, test_prompt):
        vanilla_prompt = vanilla_ICL_prompt + test_prompt
        vanilla_pred = get_response(model, tokenizer, vanilla_prompt)
        y = test_df.iloc[test_id, label_col]
        if str(y) not in vanilla_pred:
            return -1, -1
        scores = []
        killed = []

        for mutator in all_mutators:
            cnt, acc = 0, 0
            for mutated_prompt in mutated_ICL_prompts[mutator]:
                final_prompt = mutated_prompt + test_prompt
                pred = get_response(model, tokenizer, final_prompt)
                if str(y) in pred:
                    acc += 1
                    killed.append(0)
                else:
                    killed.append(1)
                cnt += 1

            scores.append(1 - acc / cnt)
        return scores, killed
    
    all_scores = []
    
    for test_id in range(args.test_example):
        test_prompt = format_example(test_df, test_id, task)
        score, killed = mutatoion_score(test_id, test_prompt)
        if score == -1:
            continue
        else:
            all_scores.append([test_id, *score])

            if len(all_scores) % 10 == 0:
                result_df = pd.DataFrame(all_scores, columns=['test_id', *all_mutators])
                result_detail = pd.DataFrame(killed, columns=['test_id', *range(len(all_mutators) * args.shots)])
                result_df.to_csv(f'./results/{args.model}_{args.task}.csv')
                result_detail.to_csv(f'./results/{args.model}_{args.task}_detail.csv')
                print(f'{len(all_scores)} cases saved. Current test id: {test_id}')
    