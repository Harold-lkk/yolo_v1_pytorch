# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-08-05 14:42:29
@LastEditors: lkk
@Description: 
'''
from datasets import load_datasets
from net import NetWork
from utils import get_num_correct
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data.distributed import DistributedSampler

tb = SummaryWriter()
net = NetWork()
optimizer = optim.Adam(net.parameters(), lr=0.01)
train_loader = load_datasets(1)
# images, labels = next(iter(train_loader))
# grid = torchvision.utils.make_grid(images)
# tb.add_image("image", grid)
# tb.add_graph(net, grid)
torch.distributed.init_process_group(backend='nccl')
# net = torch.nn.parallel.DistributedDataParallel(net)
local_rank = torch.distributed.get_rank()

epoch = 6
for e in range(epoch):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch
        preds = net(images)
        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print(
        "epoch", e,
        "total_loss", total_loss,
        "total_correct", total_correct
        )
    tb.add_scalar("loss", total_loss, e)
    tb.add_scalar("Number Correct", total_correct, e)
    # tb.add_scalar("Accuracy")
    tb.add_histogram("conv1.bias", net.conv1.bias, e)
    tb.add_histogram("conv1.weight", net.conv1.weight, e)
    tb.add_histogram("conv1.weight.grad", net.conv1.weight.grad, e)
