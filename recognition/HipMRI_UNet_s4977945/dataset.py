"""
This file is for a dataloader class that loads the nib files in the HipMRI Study
Author: Trond Jakob Grø Rein (s4977945)
"""
import numpy as np
import torch, os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import nibabel as nib

# Loading in some useful constants from utils
from utils import TRAIN_DIR, TRAIN_SEG_DIR, TEST_DIR, TEST_SEG_DIR, VAL_DIR, VAL_SEG_DIR, NUM_CLASSES

class HipMRIDataset(Dataset):
    def __init__(self,
                 img_set,
                 size,
                 num_classes=NUM_CLASSES,
                 transform=False):
        
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
        
        
        self.size = (size, size) # HxW size
        self.num_classes = num_classes
        self.transform = transform

        self.images = sorted(file for file in os.listdir(self.image_dir) if file.endswith('.nii.gz')) # Just take the nib files, in case there's some other files bundled in same folder
        self.masks = sorted(file for file in os.listdir(self.mask_dir) if file.endswith('.nii.gz'))
        
        assert len(self.images) == len(self.masks), "Image/Mask count mismatch" # Check if got masks for all images
        

    def __len__(self):
        return len(self.images) # Length (size) of the dataset is the same as images

    # Little helper function to clean up code, it is a little simplified from the code in Appendix B from assignment paper
    def _load_nii(self, path):
        nii = nib.load(path).get_fdata(caching="unchanged") 
        # If it loads as 3D, remove the third dimension
        if nii.ndim == 3:
            nii = nii[:, :, 0]
        return nii
    
    
    def __getitem__(self, idx):
        img = self._load_nii(os.path.join(self.image_dir, self.images[idx]))
        mask = self._load_nii(os.path.join(self.mask_dir, self.masks[idx]))
        
        # Apply transform, just a little z-normalization
        if self.transform:
            img = (img - img.mean()) / img.std()
        
        # Need to expand the dimension to (1,H,W) to be able to use Resize function later
        img = np.expand_dims(img, axis=0) 
        mask = np.expand_dims(mask, axis=0)
        
        # Make the images and masks tensors.
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Make them the right size
        # The image is interpolated bilinear for images, and the mask to the nearest to preserve class-IDs
        img = transforms.Resize(size=self.size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(img)
        mask = transforms.Resize(size=self.size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        
        
        mask = mask.squeeze(0) # Squeeze the mask back to (H,W) not (1,H,W), it's how the DiceLoss expects it.

        return img, mask
