# PneumoDetect

[![Fastai](https://img.shields.io/badge/Fastai-v2.5.3-blue)](https://www.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9.0-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8-green)](https://www.python.org/)
[![Psutil](https://img.shields.io/badge/Psutil-v5.8.0-yellow)](https://pypi.org/project/psutil/)

## Project Overview

This project is aimed at developing a deep learning model to detect pneumonia from chest X-ray images using the Fastai library. The project uses a convolutional neural network (CNN) based on the ResNet34 architecture and leverages data augmentation techniques and transfer learning for improved performance.

## Table of Contents

- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Making Predictions](#making-predictions)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [System Resource Monitoring](#system-resource-monitoring)
- [Contributors](#contributors)
- [License](#license)

## Setup

### Requirements

Ensure you have the following dependencies installed:
- fastai
- torch
- pandas
- numpy
- pathlib
- psutil

You can install these dependencies using pip:
```bash
pip install fastai torch pandas numpy pathlib psutil
```

### Directory Structure

Your data should be organized in the following structure:
```
/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Data Preparation

```python
from fastai.vision.all import *
import numpy as np
from pathlib import Path

np.random.seed(40)
item_tfms = Resize(224)
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
    splitter=RandomSplitter(valid_pct=0.2, seed=40)
)
dls = dblock.dataloaders(path, num_workers=4)
dls.show_batch()
```

## Training the Model

```python
import torch
from fastai.vision.all import *
import psutil
from pathlib import Path

# Monitor system resources
def log_system_resources():
    mem = psutil.virtual_memory()
    gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"CPU Memory: {mem.percent}% used, GPU Memory: {gpu_mem / 1e9:.2f} GB used")

# Paths to your data
train_path = Path('/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/train/')
valid_path = Path('/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/val/')

# Load data with a reduced batch size
batch_size = 16
data = ImageDataLoaders.from_folder(
    train_path.parent, train='train', valid='val',
    bs=batch_size, item_tfms=Resize(224)
)

# Create the learner
learn = cnn_learner(data, resnet34, metrics=accuracy)

# Initial training
lr1 = 1e-3
lr2 = 1e-1
log_system_resources()
learn.fit_one_cycle(4, slice(lr1, lr2))
log_system_resources()
```

## Fine-tuning the Model

```python
# Extended training with a single learning rate
lr = 1e-1
log_system_resources()
learn.fit_one_cycle(20, slice(lr))
log_system_resources()

# Unfreeze and find optimal learning rate
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

# Fine-tuning with lower learning rates
log_system_resources()
learn.fit_one_cycle(10, slice(1e-4, 1e-3))
log_system_resources()

# Plot the loss
learn.recorder.plot_losses()
```

## Making Predictions

```python
from fastai.vision.all import *

# Load the model
learn = load_learner('/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/model.pkl')

# Making a prediction
pred = learn.predict('/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg')
print(pred)
```

## Saving and Loading the Model

```python
# Save the model
learn.save('final_model')

# Load the model
learn = load_learner('/home/harsha/Desktop/Pneumonia/chest_xray/chest_xray/model.pkl')
```

## System Resource Monitoring

Monitoring system resources during training:
```python
import psutil
import torch

def log_system_resources():
    mem = psutil.virtual_memory()
    gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"CPU Memory: {mem.percent}% used, GPU Memory: {gpu_mem / 1e9:.2f} GB used")
```

## Contributors

- Harsha (Your Name)

## License

This project is licensed under the MIT License.
