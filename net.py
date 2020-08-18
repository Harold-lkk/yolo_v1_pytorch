# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-18 23:17:17
LastEditors: lkk
@Description: 
'''
import torch
from torch import nn as nn
from backbone import DarkNet19, DarkNetBase


class YOLOV1(nn.Module):
    def __init__(
        self,
        class_nums,
        box_nums=2,
        grid_size=7,
        lambda_coord=5,
        lambda_noobj=0.5,
        conv_mode=False
    ):
        super().__init__()
        self.backbone = DarkNet19()
        self.class_nums = class_nums
        
    def forward(self, input):
        return input


if __name__ == "__main__":
    w = 8
    h = 14
    x = torch.linspace(0, w - 1, w).unsqueeze(dim=0).repeat(h,
                                                            1).unsqueeze(dim=0)
    y = torch.linspace(0, h - 1, h).unsqueeze(dim=0).repeat(
        w, 1).unsqueeze(dim=0).permute(0, 2, 1)