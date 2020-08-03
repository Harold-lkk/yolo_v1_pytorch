# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2020-07-30 14:42:59
@LastEditTime: 2020-07-30 15:50:09
@LastEditors: lkk
@Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
print(conv.weight)