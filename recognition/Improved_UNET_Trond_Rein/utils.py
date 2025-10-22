import torch
import numpy as np

""" CONSTANTS AND PARAMS """
# setting device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# directory for files
TRAIN_DIR = './data/OASIS/keras_png_slices_train'
TRAIN_SEG_DIR = './data/OASIS/keras_png_slices_seg_train'

TEST_DIR = './data/OASIS/keras_png_slices_test'
TEST_SEG_DIR = './data/OASIS/keras_png_slices_seg_test'

VAL_DIR = './data/OASIS/keras_png_slices_validate'
VAL_SEG_DIR = './data/OASIS/keras_png_slices_seg_validate'

# some image params
IMAGE_SIZE = 128
NUM_CLASSES = 4
CLASS_VALUES = np.array([0, 85, 170, 255], dtype=np.uint8)
BATCH_SIZE = 4


# number of epochs to run
EPOCHS = 150

