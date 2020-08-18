# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-08-03 14:51:07
@LastEditors: lkk
@Description: fffff
'''

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()