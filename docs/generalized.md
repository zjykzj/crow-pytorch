
# Generalized Framework

1. Input image. Default: 224x224
2. Extract feature. Default: Layer4 for ResNet
3. Enhance feature. Default: CroW
4. Dimension reduction. Default: None
5. Compute distance. Default: Euclidean distance
6. Rank. Default: Reverse sort by distance
7. Re-Rank. Default: None
8. Eval criteria. Default: mAP for Oxford5k

## Dimension reduction

### Only L2-Norm

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 1
Loading features oxford/layer4 ...
Processing file 5000
52it [00:01, 31.93it/s]
55it [00:01, 31.85it/s]
mAP: 0.431063
```

### L2-Norm -> PCA -> L2-Norm

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 2 --whiten_features oxford/layer4 --d 1024
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=2048 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 20.34it/s]
55it [00:02, 20.30it/s]
mAP: 0.410195
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 2 --whiten_features oxford/layer4 --d 1024
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=1024 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:01, 25.60it/s]
55it [00:01, 27.96it/s]
mAP: 0.409905
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 2 --whiten_features oxford/layer4 --d 512
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=512 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
52it [00:01, 33.58it/s]
55it [00:01, 33.41it/s]
mAP: 0.407927
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 2 --whiten_features oxford/layer4 --d 256
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=256 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
55it [00:01, 46.81it/s]
55it [00:01, 46.48it/s]
mAP: 0.400525
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 2 --whiten_features oxford/layer4 --d 128
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=128 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:00, 54.73it/s]
55it [00:00, 55.18it/s]
mAP: 0.393244
```

### L2-Norm -> PCA(Whitening) -> L2-Norm

Two methods are tried, one is based on Oxford dataset, the other is based on Paris dataset

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features oxford/layer4 --d 512
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=512 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
52it [00:01, 35.78it/s]
55it [00:01, 34.87it/s]
mAP: 0.444140
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
52it [00:01, 32.78it/s]
55it [00:01, 34.21it/s]
mAP: 0.471539
```

### Brief Summary

From the above experiments, it can be found that dimension reduction can indeed further advance the performance. The best experimental configuration is `L2-Norm -> PCA(Whitening with Paris) -> L2-Norm`

## Euclidean vs. Cosine

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --dis cosine
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 22.72it/s]
55it [00:02, 24.35it/s]
mAP: 0.470833
```

## Re-Rank

The rerank step is tried, combines the first k images then sorted again

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 1
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 23.14it/s]
55it [00:02, 24.81it/s]
mAP: 0.520134
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 2
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 24.31it/s]
55it [00:02, 23.49it/s]
mAP: 0.534302
```

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
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 5
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 25.27it/s]
55it [00:02, 25.61it/s]
mAP: 0.523430
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features paris/layer4 --d 512 --qe 10
Loading features paris/layer4 ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
54it [00:02, 25.36it/s]
55it [00:02, 24.91it/s]
mAP: 0.487575
```

## Conclusion

After the above experiments, the optimal configuration is

1. Input size: `224x224`
2. Extract feature: `Layer4 for ResNet`
3. Enhance feature: `CroW`
4. Dimension reduction: `L2-Norm -> PCA(Whitening with Paris) -> L2-Norm`
5. Compute distance. Default: `Euclidean distance`
6. Rank: Default
7. Re-Rank: `QE`
8. Eval criteria. Default: mAP for Oxford5k

## 224 vs. Origin

Using the configuration debugged above, test the evaluation result of the original input image size again.

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4 --origin
python extract_features.py --images paris/data/* --out paris/layer4 --layer layer4 --origin
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4 --origin

$ python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 3 --whiten_features oxford/layer4 --d 512 --qe 3
Loading features oxford/layer4 ...
Processing file 5000
Fitting PCA/whitening wth d=512 on oxford/layer4 ...
Loading features oxford/layer4 ...
Processing file 5000
55it [00:02, 19.69it/s]
55it [00:02, 18.59it/s]
mAP: 0.613496

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

## bak

```shell
python extract_features.py --images oxford/data/* --out oxford/avgpool --layer avgpool
python extract_features.py --images paris/data/* --out paris/avgpool --layer avgpool
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer avgpool

python evaluate.py --queries oxford/avgpool_queries --groundtruth oxford/groundtruth --index_features oxford/layeravgpool --wt crow --dw 3 --whiten_features paris/avgpool --d 512 --qe 3
Loading features paris/avgpool ...
Processing file 6300
Fitting PCA/whitening wth d=512 on paris/avgpool ...
Loading features oxford/layeravgpool ...
Processing file 5000
54it [00:02, 25.43it/s]
55it [00:02, 26.55it/s]
mAP: 0.524220
```