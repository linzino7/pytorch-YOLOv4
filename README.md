# pytorch-YOLOv4
A minimal PyTorch implementation of YOLOv4.

This Rep forked from [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4). See [Original_README](https://github.com/linzino7/pytorch-YOLOv4/blob/master/Original_README.md).

## Hardware
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
- 31 RAM
- NVIDIA RTX 1080 8G * 4

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:

1.  [Dataset Preparation](#Dataset-Preparation)
2.  [Training](#Training)
3.  [Inference](#Inference)


## Pytorch Weights Download 
- google (provided by Tianxiaomo/pytorch-YOLOv4)
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

## Dataset Preparation
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Prepare Images
After downloading images, the data directory is structured as:
```
train.txt
  +- data/
  | +- train/
  | +- test/
  | +- training_labels.csv
  | +- val.txt

```

#### Download Classes Image
Data: https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl

Download and extract *tain.tar.gz* and *test.tar.gz* to *data* directory.


## Training
To train models, run following commands.
```
$ python3 train.py -d data/ -classes 10 -g 0 -pretrained ./weight/yolov4.conv.137.pth
```
The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | Bach Size |
------------ | ------------- | ------------- | ------------- | ------------- | -------------|
YOLOv4 | 1x NVIDIA T4 | 608x608 | 1 | 2.5 hours | 4 |
YOLOv4 | 4x NVIDIA RTX 1080 8G | 608x608 | 1 | 0.6 hour | 32 |

### Muti-GPU Training
```
$ python3 train.py -d data/ -classes 10 -g 0,1,2,3 -pretrained ./weight/yolov4.conv.137.pth
```

## Inference

### Inference single images
```
$ python3 models.py 10 Yolov4_epoch10.pth data/test/1.png 608 608 data/SVHN.names
```

### Inference images in folder
```
$ python3 models_mut.py 10 Yolov4_epoch22_pre.pth data/test/ 608 608 data/SVHN.names
```

# Reference:
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
