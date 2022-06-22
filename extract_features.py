# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os

import torch
from torch import Tensor
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
from PIL import Image

from resnet import resnet50


###################################
# Feature Extraction
###################################

def load_img(path: str) -> Image:
    """
    Load the image at the provided path and normalize to RGB.

    :param str path:
        path to image file
    :returns Image:
        Image object
    """
    try:
        img = Image.open(path)
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        return rgb_img
    except:
        return None


def format_img_for_vgg(img: Image, transform: transforms.Compose) -> Tensor:
    """
    Given an Image, convert to ndarray and preprocess for VGG.

    :param Image img:
        Image object
    :returns ndarray:
        3d tensor formatted for VGG
    """
    # # Get pixel values
    # d = np.array(img, dtype=np.float32)
    # d = d[:, :, ::-1]
    #
    # # Subtract mean pixel values of VGG training set
    # d -= np.array((104.00698793, 116.66876762, 122.67891434))
    #
    # return d.transpose((2, 0, 1))
    return transform(img)


def extract_raw_features(net: torch.nn.Module, layer: str, d: Tensor) -> np.ndarray:
    """
    Extract raw features for a single image.
    """
    # # Shape for input (data blob is N x C x H x W)
    # net.blobs['data'].reshape(1, *d.shape)
    # net.blobs['data'].data[...] = d
    # # run net and take argmax for prediction
    # net.forward()
    # return net.blobs[layer].data[0]

    # Reshape input from (C, H, W) to (N, C, H, W)
    data = d.reshape(1, *d.shape)
    return net(data, type=layer)[0].detach().cpu().numpy()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--images', dest='images', type=str, nargs='+', required=True,
                        help='glob pattern to image data')
    parser.add_argument('--layer', dest='layer', type=str, default='layer4',
                        choices=['layer4', 'avgpool', 'fc'],
                        help='model layer to extract')
    # parser.add_argument('--layer', dest='layer', type=str, default='pool5', help='model layer to extract')
    # parser.add_argument('--prototxt', dest='prototxt', type=str, default='vgg/VGG_ILSVRC_16_pool5.prototxt',
    #                     help='path to prototxt')
    # parser.add_argument('--caffemodel', dest='caffemodel', type=str, default='vgg/VGG_ILSVRC_16_layers.caffemodel',
    #                     help='path to model params')
    parser.add_argument('--out', dest='out', type=str, default='', help='path to save output')
    args = parser.parse_args()

    # net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net = resnet50(pretrained=True)
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    for path in tqdm(args.images):
        img = load_img(path)

        # Skip if the image failed to load
        if img is None:
            print(path)
            continue

        d = format_img_for_vgg(img, transform)
        X = extract_raw_features(net, args.layer, d)

        filename = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(args.out, filename), X)
