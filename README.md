# I Want That LOOK - YOLOX Fashion Search Enabler
It all starts and ends with the customer experience.
<p align="center">
  <img src="https://github.com/ArdaniahJ/Custom_Training_with_YOLOX_ONNX/blob/main/demo.gif" />
</p>

# Overview
`Empowering Fashion Search with YOLOX Object Detection to Enhance Fashion Recognition and Recommendation.` Featuring `YOLOX`, Fashion Search is tailored to the needs of customers dealing with a wide range of apparel (see tables below for the apparel listed). <br><br>`Fashion Search Enabler` is a garments detection detector based on YOLOX. Given a fashion image, the detector is able to finds and localize potential garments and recommend the similar or the exact garments by the end of every clicks. 

## Expected Tangible Outcome
__The goal of this project is to develop a personalized fashion search engine utilizing YOLOX _to achieve 40% accuracy and efficiency_ from client's current system, targeting to _boost gross monthly sales revenue by 15% within 6 months_ after intergratio with client's E-commerce platform__

# About Dataset
The image datasets are collected from 2 open resources:
1. Publicly available fashion dataset: [Kaggle](https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection)
2. Custom dataset from Google images and the sites of web fashion shops: [Raw External Images ](https://github.com/ArdaniahJ/Custom_Training_with_YOLOX_ONNX/blob/main/Raw%20external%20images%20for%20annotations.zip)

## Custom Image Annotations on Remo.ai
<p align="center">
  <img src="https://github.com/ArdaniahJ/Custom_Training_with_YOLOX_ONNX/blob/main/remo_ai_tagging.gif" />
</p>

The initial amount of fashion garment samples is __2682__.
With additional 10 external images that was annotated through [Remo.ai](https://remo.ai/), the number of total image samples used for this project is __2693__. 

To install Remo.ai:
```python 
pip install remo
python -m remo_app init
python -m remo_ap
```

Login/Sign Up, Upload the raw images (scraped online) and start annotating. 
Afterwards, add those annotated images in folder `JPEGImages` to become a whole complete dataset.

Finally, the image data consisted of 11 classes:
  
 Class | Label|
| :------------: |:---------------:|
| 0 | sunglass| 
| 1   | hat      |
| 2  | jacket       |
| 3 | shirt   |
| 4 | pants   |
| 5 | shorts   |
| 6 | skirt   |
| 7 | dress   |
| 8 | bag   |
| 9 | shoe   |
| 10 | top   |

## Converting from Remo JSON Format to PASCAL VOC Format (.xml)
REMO.ai gives out JSON format to export, while YOLOX takes in XML format. Hence 2 functions for the conversion are written for such action;
1. function __to specify bounding boxes in xml__ ([bbox_to_xml.py](https://github.com/ArdaniahJ/Custom_Training_with_YOLOX_ONNX/blob/main/bbox_to_xml.py))
2. function that __reads & convert the json file to xml__ ([json_to_xml.py](https://github.com/ArdaniahJ/Custom_Training_with_YOLOX_ONNX/blob/main/json_to_xml.py))

 Before run __json_to_xml.py__ ;
 1. Make a new directory with a new folder named 'annotations'
 
 ```!mkdir annotations```
 
 2. Upload the dataset.json file;
 For Colab:
 ```python
 from google.colab import files
 uploaded = files.upload()
 ```
# Installation of YOLOX Model:
```python
!git clone https://github.com/Megvii-BaseDetection/YOLOX

# Change the directory to YOLOX and download the dependencies
%cd /content/YOLOX

!pip3 install -U pip && pip3 install -r requirements.txt
!pip3 install -v -e . # or python3 setup.py develop
```
# Training
## Training Preparation
Sort the necessary files to a Pascal VOC folder for the one time json to xml conversion;
```python
!pwd # make sure to be in /content/YOLOX/dataset directory

# Create a parent (-p) folder of VOCdevkit and a subfolder of VOC2007
!mkdir -p VOCdevkit/VOC2007

# Move Annotations, ImageSets, JPEGImages folder from datasets/fashion to datasets/VOCdevkit/VOC2007
shutil.move ("/content/YOLOX/datasets/fashion/colorful_fashion_dataset_for_object_detection/Annotations", "/content/YOLOX/datasets/VOCdevkit/VOC2007")
shutil.move ("/content/YOLOX/datasets/fashion/colorful_fashion_dataset_for_object_detection/ImageSets", "/content/YOLOX/datasets/VOCdevkit/VOC2007")
shutil.move ("/content/YOLOX/datasets/fashion/colorful_fashion_dataset_for_object_detection/JPEGImages", "/content/YOLOX/datasets/VOCdevkit/VOC2007")
```

## In order to avoid error during training. 

Go to `/content/YOLOX/exps/example/yolox_voc/yolox_voc_x.py` line 46 & 102. 

1. Make sure the VOC is only VOC2007 for both. 
Remove VOC2012 if it's there to avoid confusion for the model. 
  + 🔖 **It's actually can be customized to 'Custom' or VOC2012 itself, but since we're already doing VOC2007 then leave it as it is.**
2. Next, on line 14, PASCAL VOC has 20 classes but this project only has 11. Edit it to match with fashion datasets and save. 
3. Default/max epochs for training is 300 epochs. In order to change it, go to line 14 and typed in new `self.max_epoch = 20` since we have few classes only. (This to avoid overfitting too)
4. There will be an error for 'classes' during the training. So go to `/content/YOLOX/yolox/data/datasets/voc_classes.py` and modified the classes to match with the project (those 11 classes).

### Set the training pipeline and run the training
To match with the stability of the dataset and model performance, choose: `yolox_x`for training but it is always good to experiment with other variant as well.
```python
!python './tools/train.py' -f './exps/example/yolox_voc/yolox_voc_x.py' -d 1 -b 16 --fp16 -o -c './weights/yolox_x.pth'
```

# Evaluation
Play around with the confidence score and compare the mAP:
```python
# 1. conf: 0.1
# -d:1 is 1 GPU
!python -m './tools/eval.py' -f exps/example/yolox_voc/yolox_voc_x.py -c weights/best_yolox_s.pth -b 64 -d 1 --conf 0.1 -fp16
```

# Export to ONNX
```python
# Export the model to ONNX format
```python
!python3 tools/export_onnx.py -f exps/example/yolox_voc/yolox_voc_x.py --output-name weights/best_yolox_x.onnx -c weights/best_yolox_x.pth
```

Download he model with ONNX's optimized best weights
```python
# Redirect the directory 
%cd /YOLOX/weights/

from google.colab import files
files.download('best_yolox_x.onnx') 
```

### To-do list
- [/] Build YOLOX API with FastAPI Framework - yet to upload on repo
- [ ] Deploy YOLOX API with Heroku
- [ ] Load Testing deployed YOLOX API with Locust

