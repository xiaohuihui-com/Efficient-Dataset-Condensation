import os
import datetime
import argparse
import torch
from models import model_param_init
from utils.common import set_seed
from utils.dataloader import load_data
from utils.trainer import Trainer, train_data
from config import get_config, update_config
from config.logger import Logger

logger = Logger()


def main(args):
    start_time = datetime.datetime.now()
    opt = get_config(args.config)
    update_config(opt, args.__dict__)
    save_dir = os.path.join(opt.save_dir, opt.dataset, opt.model)
    os.makedirs(save_dir, exist_ok=True)
    logger.info('{}'.format(opt))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_ids
    set_seed(opt.seed)
    opt.save_ckpt = True
    # get data
    train_loader, val_loader, nclass = load_data(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info("Using {} device.".format(device))
    logger.info('Using {} dataloader workers every process'.format(opt.num_workers))
    logger.info("Using {} {} images for training, {} images for validation." \
                .format(opt.dataset.upper(), len(train_loader.dataset), len(val_loader.dataset)))

    if opt.dataset == 'cifar100':
        opt.num_classes = 100
    elif opt.dataset in ['mnist', 'fashion']:
        opt.channel = 1
        opt.size = 28

    model, criterion, optimizer, scheduler = model_param_init(opt, device)
    train_data(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, opt, logger)

    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).total_seconds()
    logger.info('Using {} {} run time：{:.2f}h'.format(opt.dataset.upper(), opt.model, run_time / 3600.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--config", default='config.yaml')
    parser.add_argument('--dataset', type=str, \
                        default='cifar10',
                        choices=['mnist', 'fashion', 'svhn', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    main(args)
