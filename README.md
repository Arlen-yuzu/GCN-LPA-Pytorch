# GCN-LPA-PyTorch
This repository is the PyTorch Implementation of GCN-LPA ([arXiv](https://arxiv.org/abs/2002.06755)):

> Unifying Graph Convolutional Neural Networks and Label Propagation  
> Hongwei Wang, Jure Leskovec  
> arXiv Preprint, 2020

GCN-LPA is an end-to-end model that unifies Graph Convolutional Neural Networks (GCN) and Label Propagation Algorithm (LPA) for adaptive semi-supervised node classification.

## TODO
```
$ python main.py
```

## Required packages

The code has been tested running under Python 3.8.12, with the following packages installed (along with their dependencies):
- torch==1.10.1
- scipy==1.7.3
- numpy==1.21.2
- pyg==2.0.3
- pytorch-scatter==2.0.9
- pytorch-sparse==0.6.12
- pytorch-spline-conv==1.2.1

## Notification
- This is not an official implementation.
- Please cite the following papers if you use the code in your work:
```
@article{wang2020unifying,
    title={Unifying Graph Convolutional Neural Networks and Label Propagation},
    author={Hongwei Wang and Jure Leskovec},
    journal={arXiv preprint arXiv:2002.06755}
    year={2020},
}
```