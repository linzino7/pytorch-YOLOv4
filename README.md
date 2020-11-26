# pytorch-YOLOv4
A minimal PyTorch implementation of YOLOv4.

This Rep forked from [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4).

See [Original_README] (https://github.com/linzino7/pytorch-YOLOv4/blob/master/Original_README.md).

## Hardware
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
- 31 RAM
- NVIDIA RTX 1080 8G * 4

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Training](#Training)
4. [Inference](#Inference)


## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
pip install -r requirements.txt
```

## Dataset Preparation
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Prepare Images
After downloading and converting images, the data directory is structured as:
```
data
  +- training_data
  |  +- training_data
  +- testing_data
  |  +- testing_data
  +- training_labels.csv
```

#### Download Official Image
Download and extract *cs-t0828-2020-hw1.zi* to *dataw* directory.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c cs-t0828-2020-hw1
$ mkdir data
$ unzip cs-t0828-2020-hw1.zip -d data
```


## Training
To train models, run following commands.
```
$ python3 train.py -d data/ -classes 10 -g 0 -pretrained ./weight/yolov4.conv.137.pth
```
The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | Bach Size |
------------ | ------------- | ------------- | ------------- | ------------- | -------------|
YOLOv4 | 1x NVIDIA T4 | 608x608 | 1 | 2.5 hours | 4 |
YOLOv4 |4x NVIDIA RTX 1080 8G | 608x608 | 1 | 0.6 hour | 32 |

### Muti-GPU Training
```
$ python3 train.py -d data/ -classes 10 -g 0,1,2,3 -pretrained ./weight/yolov4.conv.137.pth
```

## Inference


Reference:
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
