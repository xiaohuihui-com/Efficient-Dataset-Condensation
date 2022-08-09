# Efficient-Dataset-Condensation

------

Official PyTorch implementation of **"[Dataset Condensation via Efficient Synthetic-Data Parameterization](https://arxiv.org/abs/2205.14959)"**, published at **ICML'22**

## 项目简介

------

对源仓库代码进行了重构优化，增加了log输出，减少参数设置，代码模块化，可读性更强。仅供学习使用。

## Requirements

----

- torch

- pyyaml
- matplotlib
- tqdm

## 项目文件目录

----

> Efficient-Dataset-Condensation
>  ├── condense.py       #训练生成合成图像
>  ├── config                   #log配置文件
>  │   ├── logger.py
>  │   ├── logger.yaml
>  │   ├── settings.py
>  │   └── __init__.py
>  ├── config.yaml        #参数配置文件
>  ├── evaluation.py    #评估合成图像准确率
>  ├── models               #网络模型
>  │   ├── convnet.py
>  │   ├── densenet.py
>  │   ├── loss_optim_scheduler.py
>  │   ├── resnet.py
>  │   ├── resnet_ap.py
>  │   └── __init__.py
>  ├── README.md
>  ├── requirements.txt
>  ├── train.py       #训练教师网络权重
>  ├── train.sh       #运行脚本
>  └── utils           
>      ├── augment.py
>      ├── common.py
>      ├── dataloader.py
>      ├── decoder.py
>      ├── matchloss.py
>      ├── synth.py
>      ├── trainer.py
>      ├── transformer.py
>      ├── visual.py
>      └── __init__.py

## 项目文件使用

-----

1. 克隆仓库代码： `https://github.com/xiaohuihui-com/Efficient-Dataset-Condensation.git`
2. 安装依赖包： 进入Efficient-Dataset-Condensation根目录，`pip install -r requirements.txt`

3. 训练教师网络权重

   ```python
   python train.py --dataset='cifar10' --model='convnet' --data_dir='./data'
   ```

4. 训练合成图像

   ```python
   python condense.py --dataset='cifar10' --model='convnet' --ipc=10
   ```

5. 评估合成图像 data_pt 更改为训练生成的合成图像（*.pt）

   ```python
   python evaluation.py --dataset='cifar10' --model='convnet' --ipc=10 --data_pt='data.pt'
   ```

   

## 参数说明

----

