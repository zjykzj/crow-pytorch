
# Preliminary Steps

1. Input image. Default: 224x224
2. Extract feature. Default: Layer4 for ResNet
3. Enhance feature. Default: None
4. Compute distance. Default: Euclidean distance
5. Rank. Default: Reverse sort by distance
6. Eval criteria. Default: mAP for Oxford5k

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt identity --dw 0
Loading features oxford/layer4 ...
Processing file 5000
55it [00:44,  1.26it/s]
55it [00:44,  1.23it/s]
mAP: 0.214758
```

## Which feature

Four feature extraction locations are set, which are

1. `layer4`: Convolution layer output before the last pooled layer
2. `avgpool`: Last pooled layer output (Global average pool)
3. `maxpool`: Replace the default avgpool with maxpool (Global max pool)
4. `fc`: Final classification layer output

```shell
python extract_features.py --images oxford/data/* --out oxford/avgpool --layer avgpool
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer avgpool

python evaluate.py --queries oxford/avgpool_queries --groundtruth oxford/groundtruth --index_features oxford/avgpool --wt identity --dw 0
Loading features oxford/layeravgpool ...
Processing file 5000
54it [00:01, 41.56it/s]
55it [00:01, 41.34it/s]
mAP: 0.349027
```

```shell
python extract_features.py --images oxford/data/* --out oxford/maxpool --layer maxpool
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer maxpool

python evaluate.py --queries oxford/maxpool_queries --groundtruth oxford/groundtruth --index_features oxford/maxpool --wt identity --dw 0
Loading features oxford/maxpool ...
Processing file 5000
55it [00:01, 39.98it/s]
55it [00:01, 39.61it/s]
mAP: 0.325664
```

```shell
python extract_features.py --images oxford/data/* --out oxford/fc --layer fc
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer fc

python evaluate.py --queries oxford/fc_queries --groundtruth oxford/groundtruth --index_features oxford/fc --wt identity --dw 0
Loading features oxford/fc ...
Processing file 5000
49it [00:00, 64.65it/s]
55it [00:00, 65.75it/s]
mAP: 0.309210
```

## CroW / uCroW

In the following experiments, i will try the feature enhancement method proposed in the paper.

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 0
Loading features oxford/layer4 ...
Processing file 5000
52it [00:01, 32.25it/s]
55it [00:01, 32.20it/s]
mAP: 0.349864
```

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt ucrow --dw 0
Loading features oxford/layer4 ...
Processing file 5000
55it [00:01, 39.80it/s]
55it [00:01, 39.53it/s]
mAP: 0.349027
```

From the above tests, crow achieved the best performance, but avgpool and ucrow (same as spoc) also achieved similar results

## Which size

In this paper, the original input size is recommended, because the output characteristics of convolution layer do not
need to constrain the input size. In addition, the torchvision pretraining model is based on the 224x224 input size.

```shell
python extract_features.py --images oxford/data/* --out oxford/layer4 --layer layer4 --origin
python extract_queries.py --dataset oxford --images data --groundtruth groundtruth --layer layer4 --origin

python evaluate.py --queries oxford/layer4_queries --groundtruth oxford/groundtruth --index_features oxford/layer4 --wt crow --dw 0
Loading features oxford/layer4 ...
Processing file 5000
54it [00:03, 15.81it/s]
55it [00:03, 15.86it/s]
mAP: 0.206456
```

Better results can be obtained based on the 224x224 image size, so in the next experiment, the fixed input size is 224x224.

## Conclusion

In the preliminary experiment, i test different feature extraction locations, enhance way and input size. The optimal configuration
is

1. Input: `224x224`
2. Feature location: `layer4`
3. Enhance method: `crow`