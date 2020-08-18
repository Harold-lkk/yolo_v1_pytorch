# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2020-07-30 14:42:59
@LastEditTime: 2020-08-11 16:34:28
@LastEditors: lkk
@Description: 
'''
import torch
import torch.nn as nn

import torch.nn.functional as F
torch.set_grad_enabled(True)
from collections import OrderedDict


DARKNET_CONFIG = [(32, 3, True), (64, 3, True), (128, 3, False),
                  (64, 1, False), (128, 3, True), (256, 3, False),
                  (128, 1, False), (256, 3, True), (512, 3, False),
                  (256, 1, False), (512, 3, False), (256, 1, False),
                  (512, 3, True), (1024, 3, False), (512, 1, False),
                  (1024, 3, False), (512, 1, False), (1024, 3, True)]


# def make_layers(in_channels, out_channels, kernel_size, stride, pool):
class DarkNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU(0.1)
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        if self.pool:
            x = F.max_pool2d(x)
        return x


class DarkNet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._make_layers(DARKNET_CONFIG)

    def forward(self, input):
        self.backbone(input)

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for out_channels, kernel_size, pool in DARKNET_CONFIG:
            layers.append(
                DarkNetBase(in_channels, out_channels, kernel_size,
                            pool))
            in_channels = out_channels
        return nn.Sequential(*layers)


if __name__ == '__main__':
    # from datasets import load_datasets
    # image, label = next(iter(load_datasets(1)))
    # net1 = NetWork()
    # net2 = NetWork2()
    # net1.add_module("net2", net2)
    # for name, parameter in net1.named_modules():
    #     print(name, parameter)
    # state_dict = net1.state_dict()
    # x = net1(image)
    # # print(state_dict.keys())
    # print(net1[:1])
    net = DarkNet19()
    print(net)
