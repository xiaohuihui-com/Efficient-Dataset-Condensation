import torch.nn as nn

from .convnet import ConvNet
from .resnet import ResNet
from .resnet_ap import ResNetAP
from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, densenet_cifar
from .loss_optim_scheduler import *


def get_model(s):
    return {"convnet": ConvNet,
            "resnet": ResNet,
            "resnetap": ResNetAP,
            "densenet121": DenseNet121,
            "densenet161": DenseNet161,
            "densenet169": DenseNet169,
            "densenet201": DenseNet201,
            "densenet": densenet_cifar,
            }[s.lower()]


def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'ce': ce
    }[s.lower()]


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,
    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,
    }[s.lower()]


def model_param_init(opt, device):
    net = opt.model
    if net == 'convnet':
        model = get_model(net)(num_classes=opt.num_classes, channel=opt.channel, size=opt.size,
                               **opt.model_params[net])
    elif net == 'resnet':
        model = get_model(net)(dataset=opt.dataset, num_classes=opt.num_classes, size=opt.size,
                               **opt.model_params[net])
    elif net == 'densenet':
        model = get_model(net)(nclass=opt.num_classes)
    else:
        assert 'no this networks {}'.format(net)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # get loss optimizer,schedule
    learn = opt.learning
    pg = [p for p in model.parameters() if p.requires_grad]
    criterion = get_loss(learn['loss'])().to(device)
    optimizer = get_optim(learn['optim'])(pg, **learn[learn['optim']])
    scheduler = get_scheduler(learn['scheduler'])(optimizer, milestones=[2 * opt.epochs // 3, 5 * opt.epochs // 6],
                                                  gamma=0.2)
    return model, criterion, optimizer, scheduler


def evaluation_model_list_init(opt, device):
    nets = opt.evaluation_model
    print("evaluation_model_list: {}".format(nets))
    model_list = []
    for net in nets:
        if net == 'convnet':
            model = get_model(net)(num_classes=opt.num_classes, channel=opt.channel, size=opt.size,
                                   **opt.model_params[net])
        elif net == 'resnet':
            model = get_model(net)(dataset=opt.dataset, num_classes=opt.num_classes, size=opt.size,
                                   **opt.model_params[net])
        elif net == 'densenet':
            model = get_model(net)(nclass=opt.num_classes)
        else:
            assert 'no this networks {}'.format(net)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        # get loss optimizer,schedule
        learn = opt.learning
        pg = [p for p in model.parameters() if p.requires_grad]
        criterion = get_loss(learn['loss'])().to(device)
        optimizer = get_optim(learn['optim'])(pg, **learn[learn['optim']])
        scheduler = get_scheduler(learn['scheduler'])(optimizer, milestones=[2 * opt.epochs // 3, 5 * opt.epochs // 6],
                                                      gamma=0.2)
        model_list.append((model, criterion, optimizer, scheduler))

    return model_list
