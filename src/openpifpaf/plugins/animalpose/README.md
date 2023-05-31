# openpifpaf_Traffic_Sign

## Goal of the project
The goal of this project was to implement the pifpaf architecture for the task of traffic sign detection and recognition.
We re-use openpifpaf pugglins, in our case animal pose to addapt it to our case.

The main modification in the code are in the animal-pose module where we change the keypoints used in our situation. The objective was to identify the four corners of the box as the four keypoints which are equally weighted. 

### Detection task
This is trained on the GTSDB (detection dataset). Which aims at training the detection of this modified pifpaf version.
The traffic signs are localized by an anotation file which contains bboxes of the location of the traffic sign. These annotations where scrapped and formatted at the COCO format to fit the openpifpaf requirements.

### Classification task
Then the classification of the traffic signs was supposed to be perfomed using the GTSRB (recognition dataset). This times the apprach would have been similar to the cifar 10 module of pifpaf for the training. However as we couldn't test the detection haven't tried this task. For later implemetations it should be noted that the classification doesn't perform best only using the pifpaf architecture so it could be wise to add another module to the archicteture to do the classification of the traffic signs.

### Current issues with the model and direction to solve them
The issue of the project was mainly on the use of the openpifpaf library. It had a lot of troubles we weren't able to solve over the course of the project. We followed the both the standard installation of the openpifpaf followed by the developpment for a custom dataset. 
It didn't work... We also tried to simply train the pifpaf with already existing datasets from the pifpaf website without any success.
Concerning our issue, the coco dataloader wasn't able to load any pictures which prevented the training from running. We tried to create a custom dataset even with a single image and by modifying the annotation file but there still was no successful training.



## Setup
!git clone https://github.com/openpifpaf/openpifpaf.git
pip install openpifpaf
pip install gdown
pip install scipy
pip install thop


## Dataset downloading

!pip install datasets

from datasets import load_dataset

ds = load_dataset("keremberke/german-traffic-sign-detection", name="full")

*Connect to your Drive:

from google.colab import drive
drive.mount('/content/drive')

import os
from PIL import Image

*Assuming var_1 is your list of PIL image objects

var_1 = ds['validation']['image']  # your list of PIL Image objects here

*Create a new directory in Google Drive for saving the images

new_dir = "/content/drive/My Drive/GTSDB/val"
os.makedirs(new_dir, exist_ok=True)

*Loop through the images and save each one to the new directory

for i, img in enumerate(var_1):
    img.save(os.path.join(new_dir, f'image_{i}.jpg'))


## Train
!python3 -m openpifpaf.train \
  --lr=0.0003 --momentum=0.95 --clip-grad-value=10.0 --b-scale=10.0 \
  --batch-size=16 --loader-workers=12 \
  --epochs=400 --lr-decay 360 380 --lr-decay-epochs=10 --val-interval 5 \
  --checkpoint=shufflenetv2k30 --lr-warm-up-start-epoch=250 \
  --dataset=traffic_sign --weight-decay=1e-5

## Everything else
All pifpaf options and commands still hold, please check the
[DEV guide](https://openpifpaf.github.io/dev/intro.html)
