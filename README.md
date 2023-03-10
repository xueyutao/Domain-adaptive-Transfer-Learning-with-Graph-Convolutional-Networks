## Domain-adaptive Transfer Learning with Graph Convolutional Networks

This repository contains the author's implementation in PyTorch for the paper "Coronary Heart Disease Prediction Method Fusing Domain-adaptive Transfer Learning with Graph Convolutional Networks (GCN)", Scientific Reports.


## Dependencies

- Python (>=3.6)
- Torch  (>=1.2.0)
- numpy (>=1.16.4)
- torch_scatter (>= 1.3.0)
- torch_geometric (>= 1.3.0)

## Datasets
The data folder includes different domain data. The datasets can be found in "/data/". 

## Implementation

Here we provide the implementation of DAMGCN, along with three domain datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for All-Cause, Heart Level and Mace occurs domain;
 - `dual_gnn/` contains the implementation of the Global GCN and Local GCN;

 Finally, `DAMGCN_demo.py` puts all of the above together and can be used to execute a full training run on the datasets.

## Process
 - Place the datasets in `data/`
 - Change the `dataset` in `DAMGCN_demo.py` .
 - Training/Testing:
 ```bash
 python DAMGCN_demo.py
 ```
# Citation
```
@inproceedings{xue2023DAMGCN
author={Huizhong Lin, Kaizhi Chen, Yutao Xue .etc},
title={Coronary Heart Disease Prediction Method Fusing Domain-adaptive Transfer Learning with Graph Convolutional Networks (GCN)},
journal={Scientific Reports},
year={2023}
}
```
```
Thanks to the source code provided by the author of the "https://github.com/GRAND-Lab/UDAGCN" warehouse.
```


