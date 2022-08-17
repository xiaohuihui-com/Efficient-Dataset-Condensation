# Efficient-Dataset-Condensation

论文：**[Dataset Condensation via Efficient Synthetic-Data Parameterization](https://arxiv.org/abs/2205.14959)**, **ICML'22** 

源仓库代码：**https://github.com/snu-mllab/Efficient-Dataset-Condensation**

重构优化代码：**[https://github.com/xiaohuihui-com/Efficient-Dataset-Condensation](https://github.com/xiaohuihui-com/Efficient-Dataset-Condensation)**

## 项目简介

对源仓库代码进行了重构优化，增加了log输出，减少参数设置，代码模块化，可读性更强。仅供学习使用。

## Requirements

```apl
torch
pyyaml
matplotlib
tqdm
efficientnet_pytorch
```

## About Dataset

### ImageNet数据集下载

***BT种子下载链接：https://hyper.ai/tracker/download?torrent=7144***

1. 下载Imagenet数据集BT种子。

2. 使用迅雷下载BT种子中下面文件。

   **训练集：data/ImageNet012/ILSVRC2012_img_train.tar(137G)**

   **验证集：data/ImageNet012/ILSVRC2012_img_val.tar(6.28G)**

### ImageNet数据集预处理

- 训练数据集预处理

  ILSVRC2012_img_train.tar中含有1000个类别，每个类别大约3000张图片，解压之后是已经按类别分好的文件夹。如下所示：

	```apl
--train
--------n01440764
--------n01443537
--------...
--------n15075141
	```
	
	需要找到class100.txt中所对应的100个类别。
	
- 验证数据集预处理

  ILSVRC2012_img_val.tar中含有50000张图片，解压之后是直接是图像，并没有按照类别区分开，每个类50张图片，需要我们自己去分类。

  下载*valprep.sh*脚本：[https://github.com/soumith/imagenetloader.torch](https://github.com/soumith/imagenetloader.torch)

  > 将valprep.sh 这个脚本放到val数据集中（和50000张图片放在一起）。在Linux系统中进入val目录下，输入：./valperp.sh 。如果在输入过程中发现：权限不允许，可以先输入：chmod a+x [valprep.sh](https://link.zhihu.com/?target=http%3A//valprep.sh)再输入上面的语句即可。

## 项目文件目录

```apl
- config 日志配置文件
- models 网络模型文件
- utils 工具文件
- config.yaml 默认参数配置（不需要修改）
- condense.py 训练生成合成图像pt文件
- evaluation.py 合成图像pt文件在不同网络进行评估
- train.py 训练教师网络(采用预训练时使用)
- class100.txt imagenet的100个类别
- train.sh 程序使用脚本

```

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


## 使用案列

```sh
# 训练教师网络
python train.py --dataset='cifar10' --model='convnet' 
python train.py --dataset='cifar100' --model='convnet' 
python train.py --dataset='mnist' --model='convnet' 
python train.py --dataset='fashion' --model='convnet' 
python train.py --dataset='svhn' --model='convnet' 

# 训练生成合成图像pt文件
python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=1 
python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=10 
python condense.py --dataset='cifar10' --factor=1 --init='random' --ipc=50 
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=1 
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=10 
python condense.py --dataset='cifar10' --factor=2 --init='mix' --ipc=50 

# 使用生成的合成图像pt文件在不同的网络进行评估
python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=1 --data_pt='origin_data_random_ipc1.pt' 
python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=10 --data_pt='origin_data_random_ipc10.pt' 
python evaluation.py --dataset='cifar10' --factor=1 --init='random' --ipc=50 --data_pt='origin_data_random_ipc50.pt' 
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=1 --data_pt='origin_data_ipc1.pt' 
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=10 --data_pt='origin_data_ipc10.pt' 
python evaluation.py --dataset='cifar10' --factor=2 --init='mix' --ipc=50 --data_pt='origin_data_ipc50.pt' 
```

