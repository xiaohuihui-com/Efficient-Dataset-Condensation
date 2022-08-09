#!/usr/bin/env bash


#python train.py --dataset='cifar10' --model='convnet' &
#python train.py --dataset='cifar100' --model='convnet' &
#python train.py --dataset='mnist' --model='convnet' &
#python train.py --dataset='fashion' --model='convnet' &
#python train.py --dataset='svhn' --model='convnet' &


#python condense.py --dataset='cifar10' --model='convnet' --ipc=1 &
python condense.py --dataset='cifar10' --model='convnet' --ipc=10 &
#python condense.py --dataset='cifar10' --model='convnet' --ipc=50 &

#python evaluation.py --dataset='cifar10' --model='convnet' --ipc=10 --data_pt='ipc10_iter500.pt' &
#python evaluation.py --dataset='cifar10' --model='convnet' --ipc=10 --data_pt='ipc10_iter2000.pt' &
#python evaluation.py --dataset='cifar10' --model='convnet' --ipc=50 --data_pt='ipc50_iter500.pt' &

wait