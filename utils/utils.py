import cv2
import torch
import yaml
import numpy as np


def get_yaml(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def get_lr(opt):
    lr = None
    for param_group in opt.param_groups:
        lr = param_group['lr']
    
    return lr


def prepreocess(img, size, mean=[0, 0, 0], std=[1, 1, 1]):
    mean = np.array(mean)
    std = np.array(std)

    pimg = cv2.resize(img, (size[1], size[0]))
    pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
    pimg = pimg / 255.0
    pimg = pimg - mean
    pimg = pimg / std
    pimg = np.transpose(pimg, [2, 0, 1])
    #pimg = np.expand_dims(pimg, axis=0)
    pimg = torch.from_numpy(pimg).float()

    return pimg
