# 2021-VRDL-HW3-Instance-Segmentation
HW3 Introduction: Nuclei segmentation

The proposed challenge is a nuclei segmentation, which contains two parts:
1. Transform the original train data format to coco format for detectron2 training phase
2. Segment the nuclei of bounding boxes

Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear
Train an instance segmentation model to detect and segment all the nuclei in the image 

This project uses the detectron2 pre-trained model to fix this challenge.


### Environment
- Colab
- Python Python 3.7.12
- Pytorch 1.10.0
- CUDA 10.2

### Detectron2
The project is implemented based on detectron2.
- [Detectron2](https://github.com/facebookresearch/detectron2)

## Reproducing Submission
To reproduct my submission without retrainig, run inference.ipynb on my Google Drive:
- [inference.ipynb](https://colab.research.google.com/drive/1XkynNde7CjvR3Qyx-zcOvTF6c3PB1KA_?usp=sharing)

## All steps including data preparation, train phase and detect phase
1. [Installation](#build-and-install-detectron2)
2. [Data Preparation](#dataset-preparation)
3. [Download Pretrained Model](#download-pretrained-model)
4. [Training](#training)
5. [Testing](#testing)
6. [Reference](#reference)


### Build and Install Detectron2

Run:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .

# or, as an alternative to `pip install`, use
# python setup.py build develop
```
Note: you often need to rebuild detectron2 after reinstalling PyTorch.



### Dataset Preparation
Download the given dataset from Google Drive: [train](https://drive.google.com/file/d/1FMZBGIGchY4YaUthdFeQuLNb70dI6WR5/view?usp=sharing), [test](https://drive.google.com/file/d/1KP1mguSRKBPwfSBYzKtEJBRhyT4pYbEZ/view?usp=sharing)

The files in the data folder is reorganized as below:
```
./detectron2
 ├── train_folder
 │     └── xxx
 │          ├── images - xxx.png
 │          └── masks  - masks_000x.png
 ├── test
 │     └──  yyy.png
 ├── To_Cocoformat_and_moveImg.py
 ├── submission_and_visualize.py
 └── train_Cascade.py
```

### Dataset Preprocessing
And run command `To_Cocoformat_and_moveImg.py` to create train.txt, val.txt, test.txt for training and reorganize the  data structure as below:
```
./data
 ├── train
 │     ├──  xxx.png
 │     └──  digitStruct.mat
 ├── test
 │     └──  yyy.png
 ├── dataset
 │     ├──  train.txt
 │     ├──  test.txt
 │     └──  val.txt
 ├── train.txt
 ├── test.txt
 ├── val.txt
 ├── mat_to_yolo.py
 ├── train_val_test.py
 └── shvn.yaml
```

### Download Pretrained Model
- yolov5m.pt： https://github.com/ultralytics/yolov5/releases

### Training
- to train models, run following commands.
```
python train_Cascade.py
```
After training, it may generate a folder named "output", with weight file named (iteration_count).pth

### Testing
- segment test data and creat answer.json following coco dataset format
```
python submission_and_visualize.py
```
- or download the pretrained model from Google Drive: [output](https://drive.google.com/file/d/1lmsq-2JC5aRf7a_kWp8T1b8VGfo_F1Tx/view?usp=sharing) and put 'output' in detectron2 dir

- coco dataset format

```
[{
  "image_id": image_name,
  "bbox": [[left, top, width, height]],
  "score": confidence,
  "category_id": predict_label,
  "segmentation": {
             "size": [w, h],
             "counts": RLE encoded format
             }
  
 }, 
 {
        "image_id": 1,
        "bbox": [
             734.0798950195312,
             486.75469970703125,
             28.79913330078125,
             37.0025634765625
        ],
        "score": 0.9590045213699341,
        "category_id": 1,
        "segmentation": {
             "size": [
                  1000,
                  1000
             ],
             "counts": "RX^f0c0cn05M2N2M2O2N1O1O1O001O01O00010O000O2O1O1O1N3M2M4L4M3KmmW7"
        }
 }
]
```

### Reference
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Nuclei-detection_detectron2](https://github.com/vbnmzxc9513/Nuclei-detection_detectron2)
