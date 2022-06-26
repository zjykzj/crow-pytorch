<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/zjykzj/crow-pytorch"><img align="center" src="./imgs/CroW.png"></a></div>

<p align="center">
  «crow-pytorch»使用PyTorch复现了CroW实现
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

[CroW](https://arxiv.org/abs/1512.04065)提供了一个通用的卷积特征提取框架，同时提出了无参数的空间加权和通道加权算法。另外，还提供了一个非常详细的实现框架[YahooArchive/crow](https://github.com/YahooArchive/crow)。

论文的实现是基于[caffe2](https://caffe2.ai/)，但是目前最常用的推理框架是[pytorch](http://caffe.berkeleyvision.org/)。为了能够更好的理解`CroW`实现，在论文实现的基础上使用`PyTorch`进行`CroW`算法开发。

## 安装

```shell
pip install -r requirements.txt
```

## 用法

1. 获取数据

```shell
bash oxford/get_oxford.sh
bash paris/get_paris.sh
```

2. 提前特征

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4
```

3. 编译评估工具

```shell
g++ -O compute_ap.cpp -o compute_ap
```

4. 评估实现

```shell
python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 3
```

## 主要维护人员

* Clayton Mellina - *Initial work* - [pumpikano](https://github.com/pumpikano)
* zhujian - *Enhance work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [YahooArchive/crow](https://github.com/YahooArchive/crow)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/crow-pytorch/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2022 zjykzj