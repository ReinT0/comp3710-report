"""
"dataset.py" containing the data loader for loading and preprocessing your data
"""
import numpy as np
import torch, os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from utils import TRAIN_DIR, TRAIN_SEG_DIR, TEST_DIR, TEST_SEG_DIR, VAL_DIR, VAL_SEG_DIR, NUM_CLASSES

class OasisBrainDataset(Dataset):
    def __init__(self,
                 img_set,
                 size,
                 num_classes=NUM_CLASSES):
        
        # Set which file to load given argument       
        set_map = {
            'train': (TRAIN_DIR, TRAIN_SEG_DIR),
            'test': (TEST_DIR, TEST_SEG_DIR),
            'validate': (VAL_DIR, VAL_SEG_DIR),
            'val': (VAL_DIR, VAL_SEG_DIR),
        }
        # Throw error if given wrong arg
        if img_set not in set_map:
            raise ValueError(f"img_set must be one of {list(set_map.keys())}, got {img_set!r}")

        self.image_dir, self.mask_dir = set_map[img_set]

        self.size = (size, size)
        self.num_classes = num_classes

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        
        assert len(self.images) == len(self.masks), "Image/Mask count mismatch"  
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.image_dir + "/" + self.images[idx])
        mask = Image.open(self.mask_dir + "/" + self.masks[idx])

        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5], std=[0.5])
        
        mask = np.array(mask, dtype=np.uint64)
        
        # This is for make sure the max value for a class i 255 and splits it right
        mask = np.floor(mask.astype(np.float32) * self.num_classes / 256.0).astype(np.int64)
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        mask = torch.from_numpy(mask)
        return img, mask








"""
### This is for loading Nifti files for HipMRI images
import nibabel as nib
from tqdm import tqdm

def to_channels (arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res [..., c:c+1][arr == c] = 1

    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):

    affines = []
    
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0]
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0]
        
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images
"""
