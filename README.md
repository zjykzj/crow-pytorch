<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

 <div align="center"><a title="" href="https://github.com/zjykzj/crow-pytorch"><img align="center" src="./imgs/CroW.png"></a></div>

<p align="center">
  «crow-pytorch» uses Pytorch to reproduce the CroW implementation.
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

[CroW](https://arxiv.org/abs/1512.04065) provides a general convolution feature extraction framework, and proposes parameterless spatial weighting and channel weighting algorithms. In addition, a very detailed implementation is provided - [YahooArchive/crow](https://github.com/YahooArchive/crow).

The official implementation is based on [caffe](http://caffe.berkeleyvision.org/), but the most popular deep reasoning framework at present is [pytorch](http://caffe.berkeleyvision.org/). In order to better understand the implementation of CroW, I try to replace the implementation of caffe in the warehouse with pytorch.

## Installation

```shell
pip install -r requirements.txt
```

## Usage

1. Get data

```shell
bash oxford/get_oxford.sh
bash paris/get_paris.sh
```

2. Extract features

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4
python extract_queries.py --dataset oxford
```

3. Compile eval tool

```shell
g++ -O compute_ap.cpp -o compute_ap
```

4. Evaluate

```shell
python evaluate.py --index_features oxford/layer4 --whiten_features oxford/layer4
```

## Maintainers

* Clayton Mellina - *Initial work* - [pumpikano](https://github.com/pumpikano)
* zhujian - *Enhance work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [YahooArchive/crow](https://github.com/zjykzj/crow-pytorch)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/crow-pytorch/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2022 zjykzj