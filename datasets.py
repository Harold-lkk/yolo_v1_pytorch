# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2020-07-29 10:42:24
@LastEditTime: 2020-07-29 17:10:42
@LastEditors: lkk
@Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

torch.set_printoptions(linewidth=120)

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
    batch_size=1000,
    shuffle=True
)
for sample in train_loader:
    print(1)