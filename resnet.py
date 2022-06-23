# -*- coding: utf-8 -*-

"""
@date: 2022/6/22 下午5:14
@file: resnet.py
@author: zj
@description: 
"""
from typing import Type, Union, List, Optional, Callable, Any

import torch
from torch import nn as nn, Tensor
from torchvision.models.resnet import ResNet as TResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, load_state_dict_from_url, model_urls
import torchvision.transforms as transforms

__supported_layer__ = ['layer4', 'avgpool', 'maxpool', 'fc']


class ResNet(TResNet):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)

        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor, type='layer4') -> Tensor:
        x = self._forward_impl(x)

        if type == 'layer4':
            return x
        elif type == 'avgpool':
            return self.avgpool(x)
        elif type == 'maxpool':
            return self.maxpool2(x)
        elif type == 'fc':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            raise ValueError(f'{type} does not support')


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


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


if __name__ == '__main__':
    m = resnet50()
    print(m)

    data = torch.randn(1, 3, 224, 224)
    res = m(data, type='layer4')
    assert res.shape == (1, 2048, 7, 7)
    print(res.shape)

    res = m(data, type='avgpool')
    assert res.shape == (1, 2048, 1, 1)
    print(res.shape)

    res = m(data, type='maxpool')
    assert res.shape == (1, 2048, 4, 4)
    print(res.shape)

    res = m(data, type='fc')
    assert res.shape == (1, 1000)
    print(res.shape)
