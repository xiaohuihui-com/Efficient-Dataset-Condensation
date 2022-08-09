import os
import datetime
import argparse
import torch
from utils.common import set_seed
from utils.dataloader import MultiEpochsDataLoader, load_data_path
from utils.trainer import test_data
from models import model_param_init, evaluation_model_list_init
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
    opt.load_memory = False
    # get data
    train_dataset, val_dataset = load_data_path(opt, opt.data_pt)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         persistent_workers=True)
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=opt.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    logger.info("Using {} device.".format(device))
    logger.info('Using {} dataloader workers every process'.format(opt.num_workers))
    logger.info("Using {} {} images for training, {} images for validation." \
                .format(opt.dataset.upper(), len(train_loader.dataset), len(val_loader.dataset)))

    opt.epochs = 1000
    nclass = opt.num_classes
    # model, criterion, optimizer, scheduler = model_param_init(opt, device)
    # test_data(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, opt, logger)
    model_list = evaluation_model_list_init(opt, device)
    for i, (model, criterion, optimizer, scheduler) in enumerate(model_list):
        logger.info("Using evaluating model: {}".format(opt.evaluation_model[i]))
        test_data(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, opt, logger)

    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).total_seconds()
    logger.info('Using {} {} run time：{:.2f}h'.format(opt.dataset.upper(), opt.model, run_time / 3600.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config.yaml')
    parser.add_argument('--dataset', type=str, \
                        default='cifar10',
                        choices=['mnist', 'fashion', 'svhn', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str)
    parser.add_argument('--factor', type=int)
    parser.add_argument('--ipc', type=int)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--data_pt', type=str, default='data.pt')

    args = parser.parse_args()
    main(args)
