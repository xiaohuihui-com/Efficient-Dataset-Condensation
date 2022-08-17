import os
import datetime
import argparse
import torch
from models import model_param_init, get_optim, evaluation_model_list_init
from utils.common import set_seed, save_img
from utils.matchloss import matchloss
from utils.dataloader import load_resized_data, ClassMemDataLoader, ClassDataLoader
from utils.augment import diffaug
from utils.synth import Synthesizer
from utils.trainer import train_epoch
from config import get_config, update_config
from config.logger import Logger

logger = Logger()


# 自己只需要定义:factor=1,2,3  init=mix/random  ipc  dataset
def default_params(opt):
    if opt.dataset == 'imagenet':
        opt.load_memory = True
        opt.size = 224
        opt.augment = True
        opt.n_data = 500
        opt.epochs = 2000
        opt.metric = 'l1'
        opt.mix_p = 1.0
        opt.model = 'resnetap'
        if opt.ipc == 1:
            opt.lr_img = 0.0003
        elif opt.ipc == 10:
            opt.lr_img = 0.003
        else:
            opt.lr_img = 0.006
    else:
        opt.load_memory = True
        opt.size = 32
        opt.augment = False
        opt.n_data = 2000
        opt.epochs = 1000
        opt.metric = 'mse'
        opt.mix_p = 0.5
        opt.model = 'convnet'
        if opt.ipc == 1:
            opt.lr_img = 0.0005
        elif opt.ipc == 10:
            opt.lr_img = 0.005
        else:
            opt.lr_img = 0.025
    return opt


def main(args):
    start_time = datetime.datetime.now()
    opt = get_config(args.config)
    update_config(opt, args.__dict__)
    save_dir = os.path.join(opt.save_dir, opt.dataset, opt.model)
    os.makedirs(save_dir, exist_ok=True)
    logger.info('{}'.format(opt))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_ids
    set_seed(opt.seed)
    opt = default_params(opt)
    # get data
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    trainset, val_loader = load_resized_data(opt)
    if opt.load_memory:
        train_loader = ClassMemDataLoader(trainset, batch_size=opt.batch_size, device=device)
    else:
        train_loader = ClassDataLoader(trainset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)
    nclass = opt.num_classes

    logger.info("Using {} device.".format(device))
    logger.info('Using {} dataloader workers every process'.format(opt.num_workers))
    logger.info("Using {} {} images for training, {} images for validation." \
                .format(opt.dataset.upper(), len(train_loader.dataset), len(val_loader.dataset)))
    synset = Synthesizer(opt, device)
    synset.init(train_loader, init_type=opt.init)
    save_img(os.path.join(save_dir, f'ipc{opt.ipc}_init.png'), synset.data,
             unnormalize=False,
             dataname=opt.dataset)

    aug, aug_rand = diffaug(opt, device)
    save_img(os.path.join(save_dir, f'ipc{opt.ipc}_aug.png'), aug(synset.sample(0, max_size=opt.batch_syn_max)[0]),
             unnormalize=True,
             dataname=opt.dataset)

    # model, criterion, optimizer, scheduler = model_param_init(opt, device)
    # synset.test(opt, model, val_loader, nclass, criterion, optimizer, scheduler, device, logger, bench=False)

    learn = opt.learning
    optim_img = get_optim(learn['optim'])(synset.parameters(), lr=opt.lr_img, momentum=opt.mom_img)
    niter = opt.niter
    it_test = [niter / 10, niter / 5, niter / 2, niter]  # 50，100，250，500
    logger.info(f"\nStart condensing with {opt.match} matching for {opt.niter} iteration")

    for it in range(opt.niter):
        model, criterion, optimizer, scheduler = model_param_init(opt, device)
        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        for ot in range(opt.inner_loop):
            # Update synset
            for c in range(nclass):
                img, lab = train_loader.class_sample(c)
                img_syn, lab_syn = synset.sample(c, max_size=128)
                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                loss = matchloss(opt, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                loss_total += loss.item()
                optim_img.zero_grad()
                loss.backward()
                optim_img.step()

            # Net update
            if opt.n_data > 0:
                for _ in range(1):
                    train_epoch(opt,
                                model,
                                train_loader,
                                opt.num_classes,
                                criterion,
                                optimizer,
                                device,
                                ot + 1,
                                opt.n_data,
                                aug=aug_rand,
                                mixup=opt.mixup_net)
        if it % 10 == 0:
            logger.info(f"(Iter {it:3d}) loss: {loss_total / nclass / opt.inner_loop:.2f}")
        if (it + 1) in it_test:
            save_img(os.path.join(save_dir, f'ipc{opt.ipc}_img{it + 1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=opt.dataset)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(save_dir, f'ipc{opt.ipc}_data.pt'))
            print("img and data saved!")
            # model_list = evaluation_model_list_init(opt, device)
            # synset.test(opt, model, val_loader, nclass, criterion, optimizer, scheduler, device, logger, model_list,
            #             bench=True)

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
    parser.add_argument('--init', type=str, default='mix', choices=['mix', 'random'])
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--factor', type=int, default=2)

    args = parser.parse_args()
    main(args)
