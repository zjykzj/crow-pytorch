# -*- coding: utf-8 -*-

"""
@date: 2022/6/27 上午11:32
@file: transform.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


def get_transform(origin=False):
    if origin:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return transform
