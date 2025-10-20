"""
"modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
# Loading all the nessecary libraries //
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import random

# Use GPU (CUDA-cores, much faster) if available, else use the CPU (Really slow) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        super().__init__()

        # Encoder

        # Decoder
