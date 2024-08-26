# MILE: A Mutation Testing Framework of In-Context Learning Systems
w/ Zeming Wei, Yihao Zhang, and Meng Sun.

## Quick start
1. Download `SST2, AGnews, mrpc, QNLI, RTE, WMT` datasets and move them into the folder `data`. You can directly copy the `data` folder from [BatchICL](https://github.com/Cardinalere/Batch-ICL).

2. Edit the paths to your LLMs in `paths.py`.

3. Calculate the accuracy with `eval_acc.py`. Example:
```
python eval_acc.py --model vicuna --task all --shots 20 --test-example 250
```

4. Calculate instance-wise mutation scores with `main.py`. Example:
```
python main.py --model vicuna --mutants 20 --test-example 250 --shots 20 --task SST2
```


## Acknowledgement
The data and code for formating prompts are copied from [BatchICL](https://github.com/Cardinalere/Batch-ICL).
