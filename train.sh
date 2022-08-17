#!/usr/bin/env bash


python train.py --dataset='cifar10' --model='convnet' &
python train.py --dataset='cifar100' --model='convnet' &
python train.py --dataset='mnist' --model='convnet' &
python train.py --dataset='fashion' --model='convnet' &
python train.py --dataset='svhn' --model='convnet' &


python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=1 &
python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=10 &
python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=50 &
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=1 &
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=10 &
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=50 &


python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=1 --data_pt='origin_data_random_ipc1.pt' &
python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=10 --data_pt='origin_data_random_ipc10.pt' &
python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=50 --data_pt='origin_data_random_ipc50.pt' &
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=1 --data_pt='origin_data_ipc1.pt' &
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=10 --data_pt='origin_data_ipc10.pt' &
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=50 --data_pt='origin_data_ipc50.pt' &


wait