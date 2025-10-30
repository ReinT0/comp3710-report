"""
"predict.py" showing example usage of your trained model. Print out any results and / or provide visu-
alisations where applicable
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import HipMRIDataset
from modules import ImprovedUNet
from utils import IMAGE_SIZE, NUM_CLASSES, DEVICE

device = DEVICE

def show_predictions(model, dataset, title="Segmentation Results", n=3):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            x = image.unsqueeze(0).to(device).float()
            pred = model(x)

            logits = pred[0]
            pred_labels = torch.argmax(logits, dim=0).cpu().numpy()
            pred_show = pred_labels

            img_np = image[0].cpu().numpy()
            true_mask_np = true_mask.cpu().numpy()
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            axes[1, i].imshow(true_mask_np)
            axes[1, i].set_title(f'True mask {i+1}', fontweight='bold')
            axes[1, i].axis('off')
            
            accuracy = np.mean(pred_show == true_mask_np)
            axes[2, i].imshow(pred_show)
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.2f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

ts = HipMRIDataset(img_set='test', size=IMAGE_SIZE, num_classes=NUM_CLASSES, transform=True)
best = torch.load("./outputs/best.pt")#, map_location=device)
model = ImprovedUNet().to(device)
model.load_state_dict(best['model_state'])
show_predictions(model, ts)
