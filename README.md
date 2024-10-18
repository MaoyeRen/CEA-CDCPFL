# Personalized federated learning: A Clustered Distributed Co-Meta-Learning approach
Official code for the manuscript:

Personalized federated learning: A Clustered Distributed Co-Meta-Learning approach


## Requirments
* Python3
* Pytorch
* Torchvision

## Data
* Cifar-10.
* Mimiimagenet
* Cifar-100.

## Running the experiments on cpu or gpu
* To run the  experiment  using CPU:
```
python main.py --gpu=None 
```
* Or to run it on GPU 0:
```
python main.py --gpu=0
```

## Running the experiments

* To run the  experiment  with MiniImagenet using gpu:
```
python main.py --model=vgg11 --dataset=miniimagenet --gpu=0 --cvniid=0 --useriid=1 --full_class_fill=1 --epochs=20000 --frac_cvns=1 --frac_users=0.1 --frac_train_support=0.1 --frac_train_query=0.1 --optimizer=sgd   --test_every=20000 --num_cvn=5 --num_each_cvn_user=[20]*5 --percentage=0.5 --update_lr=0.2 --meta_lr=0.01 --cvn_epoch=2 --num_classes=100
```
## Pre-steps of using other datasets or distributions
* Firstly using any of the federated cluster methods to get the clusters of clients
* Secondly writing the code to input the clusters of the clients to virtual cluster nodes, in "sampling.py".

## Cite
If you find our code useful for your research and applications, please cite us using this BibTeX:
```bibtex
@article{ren2023personalized,
title={Personalized federated learning: A Clustered Distributed Co-Meta-Learning approach},
author={Ren, Maoye and Wang, Zhe and Yu, Xinhai},
journal={Information Sciences},
volume={647},
pages={119499},
year={2023},
publisher={Elsevier}
}
```
