from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor ** 2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        elif decode_type == 'bound':
            data, target = decode_zoom_bound(data, target, factor, bound=bound)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode_zoom_bound(self, img, target, factor_max, bound=128):
    """Uniform multi-formation with bounded number of synthetic data
    """
    bound_cur = bound - len(img)
    budget = len(img)

    data_multi = []
    target_multi = []

    idx = 0
    decoded_total = 0
    for factor in range(factor_max, 0, -1):
        decode_size = factor ** 2
        if factor > 1:
            n = min(bound_cur // decode_size, budget)
        else:
            n = budget

        decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

        idx += n
        budget -= n
        decoded_total += n * decode_size
        bound_cur = bound - decoded_total - budget

        if budget == 0:
            break

    data_multi = torch.cat(data_multi)
    target_multi = torch.cat(target_multi)
    return data_multi, target_multi


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    return data_dec, target_dec
