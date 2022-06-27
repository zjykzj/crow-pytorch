# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import glob

from tqdm import tqdm
import numpy as np
from PIL import Image

from transform import get_transform
from resnet import resnet50, __supported_layer__
from extract_features import format_img_for_vgg, extract_raw_features


def query_images(groundtruth_dir, image_dir, dataset, cropped=True):
    """
    Extract features from the Oxford or Paris dataset.

    :param str groundtruth_dir:
        the directory of the groundtruth files (which includes the query files)
    :param str image_dir:
        the directory of dataset images
    :param str dataset:
        the name of the dataset, either 'oxford' or 'paris'
    :param bool cropped:
        flag to optionally disable cropping

    :yields Image img:
        the Image object
    :yields str query_name:
        the name of the query
    """
    for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
        query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        img_name, x, y, w, h = open(f).read().strip().split(' ')

        if dataset == 'oxford':
            img_name = img_name.replace('oxc1_', '')
        img = Image.open(os.path.join(image_dir, '%s.jpg' % img_name))

        if cropped:
            x, y, w, h = map(float, (x, y, w, h))
            box = map(lambda d: int(round(d)), (x, y, x + w, y + h))
            img = img.crop(box)

        yield img, query_name


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help='dataset to extract queries for')
    parser.add_argument('--images', dest='images', type=str, default='data/', help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='groundtruth/',
                        help='directory containing groundtruth files')

    parser.add_argument('--layer', dest='layer', type=str, default='layer4',
                        choices=__supported_layer__,
                        help='Model layer to extract')
    parser.add_argument('--origin', action='store_true', default=False, help='Use original input size. Default: False')
    args = parser.parse_args()
    # print('args:\n', args)

    images_dir = os.path.join(args.dataset, args.images)
    groundtruth_dir = os.path.join(args.dataset, args.groundtruth)
    out_dir = os.path.join(args.dataset, f'{args.layer}_queries')

    # Load networks
    net = resnet50(pretrained=True)
    net.eval()

    transform = get_transform(args.origin)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for img, name in tqdm(query_images(groundtruth_dir, images_dir, args.dataset)):
        d = format_img_for_vgg(img, transform)
        X = extract_raw_features(net, args.layer, d)

        np.save(os.path.join(out_dir, '%s' % name), X)
