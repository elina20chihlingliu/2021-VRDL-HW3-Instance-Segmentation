# 2021-VRDL-HW3-Instance-Segmentation
HW3 Introduction: Nuclei segmentation

The proposed challenge is a nuclei segmentation, which contains two parts:
1. Transform the original train data format to coco format for detectron2 training phase
2. Segment the nuclei of bounding boxes

Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear /n
Train an instance segmentation model to detect and segment all the nuclei in the image 

This project uses the detectron2 pre-trained model to fix this challenge.


### Environment
- Colab
- Python 3.7
- Pytorch 1.10.0
- CUDA 10.2

### YOLOv5
The project is implemented based on yolov5.
- [YOLOv5](https://github.com/ultralytics/yolov5)

## Reproducing Submission
To reproduct my submission without retrainig, run inference.ipynb on my Google Drive:
- [inference.ipynb](https://drive.google.com/file/d/14IUxba_Tjaw3teusvljHuXGmZ8rEvH1a/view?usp=sharing)

## All steps including data preparation, train phase and detect phase
1. [Installation](#install-packages)
2. [Data Preparation](#data-preparation)
3. [Set Configuration](#set-configuration)
4. [Download Pretrained Model](#download-pretrained-model)
5. [Training](#training)
6. [Testing](#testing)
7. [Reference](#reference)
