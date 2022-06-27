
# README

## Experiment

* First, i try a simple framework by convolutional feature, see [Preliminary Steps](preliminary.md)

* Then, give a generalized framework described in paper and make detailed experiments, see [Generalized Framework](generalized.md)

## Score

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 3
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 25.45it/s]
55it [00:02, 25.23it/s]
mAP: 0.547134
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4 --origin
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4 --origin
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4 --origin

$ python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features oxford/layer4 --d 512 --qe 10
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=512 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
53it [00:02, 20.63it/s]
55it [00:02, 21.92it/s]
mAP: 0.635490
```