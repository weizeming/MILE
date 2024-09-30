# MILE: A Mutation Testing Framework of In-Context Learning Systems (SETTA 2024)
[Zeming Wei](https://weizeming.github.io), Yihao Zhang, and Meng Sun.

Accepted by SETTA 2024. Preprint: https://arxiv.org/abs/2409.04831

## Usage
1. Download `SST2, AGnews, mrpc, QNLI, RTE, WMT` datasets and move them into the folder `./data`. You can directly copy the `data` folder from [BatchICL](https://github.com/Cardinalere/Batch-ICL).

2. Edit the paths to your LLMs in `paths.py`.

3. Calculate the accuracy with `eval_acc.py`. Example:
```
python eval_acc.py --model vicuna --task all --shots 20 --test-example 250
```

4. Create folder `./results` and run the mutation testing with `main.py`. The log will be saved in `./results`. Example:
```
python main.py --model vicuna --mutants 20 --test-example 250 --shots 20 --task SST2
```

5. Calculate Standard and Group-wise Mutation Scores with `analysis.py` and `mutator_analysis.py` (complete log for all models and tasks required). Example:
```
python analysis.py --num 50
python mutator_analysis.py
```
## Citation
```
@InProceedings{wei2024mile,
    title     = {MILE: A Mutation Testing Framework of In-Context Learning Systems},
    author    = {Wei, Zeming and Zhang, Yihao and Sun, Meng},
    booktitle = {SETTA},
    year      = {2024}
}
```
