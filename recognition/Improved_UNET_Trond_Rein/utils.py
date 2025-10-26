import torch
import numpy as np

""" CONSTANTS AND PARAMS """
# setting device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# directory for files
ROOT_DIR = './data/OASIS'

TRAIN_DIR = ROOT_DIR + '/keras_png_slices_train'
TRAIN_SEG_DIR = ROOT_DIR+ '/keras_png_slices_seg_train'

TEST_DIR = ROOT_DIR + '/keras_png_slices_test'
TEST_SEG_DIR = ROOT_DIR + '/keras_png_slices_seg_test'

VAL_DIR = ROOT_DIR + '/keras_png_slices_validate'
VAL_SEG_DIR = ROOT_DIR + '/keras_png_slices_seg_validate'

# some image params
IMAGE_SIZE = 256
CLASS_VALUES = np.array([0, 85, 170, 255], dtype=np.uint8)
NUM_CLASSES = len(CLASS_VALUES)
print(NUM_CLASSES)
BATCH_SIZE = 8


# number of epochs to run
EPOCHS = 5

