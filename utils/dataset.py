import os
import cv2
import torch
import numpy as np
from .utils import prepreocess
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self, base_dir, size, mean, std, mark='train') -> None:
        super().__init__()
        self.size = size
        self.mean = mean
        self.std = std

        if mark == 'train':
            img_fpath = 'trainImages.txt'
            label_fpath = 'trainLabels.txt'
        else:
            img_fpath = 'testImages.txt'
            label_fpath = 'testLabels.txt'

        img_fpath = os.path.join(base_dir, img_fpath)
        label_fpath = os.path.join(base_dir, label_fpath)
    
        with open(img_fpath, 'r') as f:
            self.img_paths = [os.path.join(base_dir, p).strip() for p in f.readlines()]
        with open(label_fpath, 'r') as f:
            self.mask_paths = [os.path.join(base_dir, p).strip() for p in f.readlines()]
        
        print("{} data length: {}".format(mark, self.__len__()))
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = cv2.imread(img_path)
        img = prepreocess(img, self.size, self.mean, self.std)

        mask = cv2.imread(mask_path)[..., 0]
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        mask[mask == 255] = 19

        return img, torch.from_numpy(mask).long()

    def __len__(self):
       return len(self.img_paths)
