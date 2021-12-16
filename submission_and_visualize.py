# coding: utf-8
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import os
from os import listdir
from detectron2.data.datasets import register_coco_instances
import pandas as pd
import os
from detectron2.utils.visualizer import ColorMode
from itertools import groupby
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion
from detectron2 import model_zoo
import json 
setup_logger()


register_coco_instances("my_dataset", {}, "nucleus_cocoformat.json", "./train")
metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "./output"  # output weight directroy path
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.DATASETS.TRAIN = ("my_dataset",)  # use training data
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("my_dataset")
predictor = DefaultPredictor(cfg)


# select image for visualizer to check our dataset weather correct
ls1 = ['TCGA-50-5931-01Z-00-DX1.png', 'TCGA-A7-A13E-01Z-00-DX1.png', 'TCGA-AY-A8YK-01A-01-TS1.png', 'TCGA-G2-A2EK-01A-02-TSB.png', 'TCGA-G9-6336-01Z-00-DX1.png', 'TCGA-G9-6348-01Z-00-DX1.png']
for name in ls1:
  im = cv2.imread("/content/detectron2/test/"+name)
  outputs = predictor(im)

  v = Visualizer(im[:, :, ::-1],
                metadata=metadata,
                scale=1,
                # remove the colors of unsegmented pixels
                instance_mode=ColorMode.IMAGE_BW
                )
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  plt.imshow(v.get_image())
  print('ok')
  out = 'demo_'+name
  cv2.imwrite(out, v.get_image())


# Submission

path = "./output/"  # the weight save path
fold = os.listdir(path)

inpath = "./test/"  # test data path

images_name = listdir(inpath)


def imdict():
  with open(r'/content/test_img_ids.json') as f:
      data = json.load(f)
  im_na = []
  im_id = []
  for im in data:
      im_na.append(im['file_name'])
      im_id.append(im['id'])
  dictionary = dict(zip(im_na, im_id))
  return dictionary


import pycocotools._mask as _mask

def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(np.asfortranarray(bimask))
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(np.asfortranarray(bimask).reshape((h, w, 1), order='F'))[0]


image_idict = imdict()
data = []

for name in images_name:
  if name[-3:]=='png':
    image = cv2.imread(inpath + name)
    outputs = predictor(image)
    im_id = image_idict.get(name)
    
    masks = np.asarray(outputs["instances"].to('cpu')._fields['pred_masks'])
    scores = outputs["instances"].to('cpu')._fields['scores'].numpy()
    Box = outputs["instances"].pred_boxes
    allins = Box.tensor.cpu()
    allins = allins.numpy()
    
    for i in range (len(allins[:,0])):
        a = {"image_id": 0, "bbox":[], "score": 0, "category_id": 1, "segmentation":{}}
        a['image_id'] = im_id
    
        instance = allins[i,:]
        left = float(instance[0])
        top = float(instance[1])
        right = float(instance[2])
        bottom = float(instance[3])
        
        width = max(0, right-left)
        height = max(0, bottom-top)

        a['bbox'] = [left, top, width, height]
        a['score'] = float(scores[i])
              
        r = {'size':[], 'counts': ''}
        RS = encode(masks[i])
        r['size'] = RS['size']
        r['counts'] = RS['counts'].decode(encoding='utf-8')
        a['segmentation'] = r
        
        # print(a)
        data.append(a)             
            
        
        
import json
ret = json.dumps(data, indent=5)
with open('/content/answer.json', 'w') as fp:
    fp.write(ret)
fp.close()
print('answer.json is fin.')
