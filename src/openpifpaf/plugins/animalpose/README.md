# openpifpaf_Traffic_Sign


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

Connect to your Drive:
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
