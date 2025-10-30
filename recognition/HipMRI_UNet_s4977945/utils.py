"""
utils.py — Global configuration (paths, device, common hyperparameters)

Keeps train/predict scripts consistent (same IMAGE_SIZE, NUM_CLASSES, device,
and data roots). Changing one value here updates the whole project.
Also make debugging easier and faster.
Author: Trond Jakob Grø Rein (s4977945)
"""

import torch
import numpy as np

""" CONSTANTS AND PARAMS """
# Compute device: prefer CUDA when available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Root dataset directory layout (HipMRI slices prepared as files)
ROOT_DIR = './data/HipMRI_Study_open/keras_slices_data'

TRAIN_DIR = ROOT_DIR + '/keras_slices_train'
TRAIN_SEG_DIR = ROOT_DIR+ '/keras_slices_seg_train'

TEST_DIR = ROOT_DIR + '/keras_slices_test'
TEST_SEG_DIR = ROOT_DIR + '/keras_slices_seg_test'

VAL_DIR = ROOT_DIR + '/keras_slices_validate'
VAL_SEG_DIR = ROOT_DIR + '/keras_slices_seg_validate'

# Input image size fed to the network (resize to H=W=256)
IMAGE_SIZE = 256

# Number of semantic classes (0=background + foreground classes)
NUM_CLASSES = 6

# Default training hyperparameters (override via CLI in train.py if needed)

# Use a higher size if using a GPU with alot of memory
BATCH_SIZE = 8

# Number of epochs to run
EPOCHS = 40

