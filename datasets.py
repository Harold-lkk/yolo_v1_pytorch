# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2020-07-29 10:42:24
@LastEditTime: 2020-08-03 14:33:53
@LastEditors: lkk
@Description: 
'''
import torch
import torchvision
import torchvision.transforms as transforms


def load_datasets(batch_size):
    train_set = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
                transforms.ToTensor()
            ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader
