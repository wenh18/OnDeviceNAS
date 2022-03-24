# (Tensorflow) EfficientNet

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a *compound coefficient*. Unlike conventional practice that arbitrary scales  these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $\alpha ^ N$,  width by $\beta ^ N$, and image size by $\gamma ^ N$, where $\alpha, \beta, \gamma$ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a  principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to [squeeze-and-excitation blocks](https://paperswithcode.com/method/squeeze-and-excitation-block).

The weights from this model were ported from [Tensorflow/TPU](https://github.com/tensorflow/tpu).

{% include 'code_snippets.md' %}

## How do I train this model?

You can follow the [mytimm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.

## Citation

```BibTeX
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
Type: model-index
Collections:
- Name: TF EfficientNet
  Paper:
    Title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
    URL: https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
Models:
- Name: tf_efficientnet_b0
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 488688572
    Parameters: 5290000
    File Size: 21383997
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: TPUv3 Cloud TPU
    ID: tf_efficientnet_b0
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.875'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '224'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1241
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 76.85%
      Top 5 Accuracy: 93.23%
- Name: tf_efficientnet_b1
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 883633200
    Parameters: 7790000
    File Size: 31512534
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b1
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.882'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '240'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1251
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.84%
      Top 5 Accuracy: 94.2%
- Name: tf_efficientnet_b2
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 1234321170
    Parameters: 9110000
    File Size: 36797929
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b2
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.89'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '260'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1261
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.07%
      Top 5 Accuracy: 94.9%
- Name: tf_efficientnet_b3
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 2275247568
    Parameters: 12230000
    File Size: 49381362
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b3
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.904'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '300'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1271
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 81.65%
      Top 5 Accuracy: 95.72%
- Name: tf_efficientnet_b4
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 5749638672
    Parameters: 19340000
    File Size: 77989689
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    Training Resources: TPUv3 Cloud TPU
    ID: tf_efficientnet_b4
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.922'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '380'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1281
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.03%
      Top 5 Accuracy: 96.3%
- Name: tf_efficientnet_b5
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 13176501888
    Parameters: 30390000
    File Size: 122403150
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b5
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.934'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '456'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1291
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 83.81%
      Top 5 Accuracy: 96.75%
- Name: tf_efficientnet_b6
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 24180518488
    Parameters: 43040000
    File Size: 173232007
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b6
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.942'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '528'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1301
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.11%
      Top 5 Accuracy: 96.89%
- Name: tf_efficientnet_b7
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 48205304880
    Parameters: 66349999
    File Size: 266850607
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b7
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.949'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '600'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1312
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 84.93%
      Top 5 Accuracy: 97.2%
- Name: tf_efficientnet_b8
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 80962956270
    Parameters: 87410000
    File Size: 351379853
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - Label Smoothing
    - RMSProp
    - Stochastic Depth
    - Weight Decay
    Training Data:
    - ImageNet
    ID: tf_efficientnet_b8
    LR: 0.256
    Epochs: 350
    Crop Pct: '0.954'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '672'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1323
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 85.35%
      Top 5 Accuracy: 97.39%
- Name: tf_efficientnet_el
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 9356616096
    Parameters: 10590000
    File Size: 42800271
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_el
    Crop Pct: '0.904'
    Image Size: '300'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1551
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 80.45%
      Top 5 Accuracy: 95.17%
- Name: tf_efficientnet_em
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 3636607040
    Parameters: 6900000
    File Size: 27933644
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_em
    Crop Pct: '0.882'
    Image Size: '240'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1541
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 78.71%
      Top 5 Accuracy: 94.33%
- Name: tf_efficientnet_es
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 2057577472
    Parameters: 5440000
    File Size: 22008479
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Data:
    - ImageNet
    ID: tf_efficientnet_es
    Crop Pct: '0.875'
    Image Size: '224'
    Interpolation: bicubic
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1531
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 77.28%
      Top 5 Accuracy: 93.6%
- Name: tf_efficientnet_l2_ns_475
  In Collection: TF EfficientNet
  Metadata:
    FLOPs: 217795669644
    Parameters: 480310000
    File Size: 1925950424
    Architecture:
    - 1x1 Convolution
    - Average Pooling
    - Batch Normalization
    - Convolution
    - Dense Connections
    - Dropout
    - Inverted Residual Block
    - Squeeze-and-Excitation Block
    - Swish
    Tasks:
    - Image Classification
    Training Techniques:
    - AutoAugment
    - FixRes
    - Label Smoothing
    - Noisy Student
    - RMSProp
    - RandAugment
    - Weight Decay
    Training Data:
    - ImageNet
    - JFT-300M
    Training Resources: TPUv3 Cloud TPU
    ID: tf_efficientnet_l2_ns_475
    LR: 0.128
    Epochs: 350
    Dropout: 0.5
    Crop Pct: '0.936'
    Momentum: 0.9
    Batch Size: 2048
    Image Size: '475'
    Weight Decay: 1.0e-05
    Interpolation: bicubic
    RMSProp Decay: 0.9
    Label Smoothing: 0.1
    BatchNorm Momentum: 0.99
    Stochastic Depth Survival: 0.8
  Code: https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/mytimm/models/efficientnet.py#L1509
  Weights: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth
  Results:
  - Task: Image Classification
    Dataset: ImageNet
    Metrics:
      Top 1 Accuracy: 88.24%
      Top 5 Accuracy: 98.55%
-->