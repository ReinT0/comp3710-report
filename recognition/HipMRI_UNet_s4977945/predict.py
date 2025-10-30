"""
predict.py - evaluation / visualization

- Loads the trained checkpoint from train.py and builds the test dataset with the SAME
  preprocessing as training (transform=True).
- Runs n examples and saves a figure in same folder as other plots:
  (image, ground truth mask, predicted mask)
- Uses argmax over channel logits to get discrete labels (multiclass seg).
Author: Trond Jakob Grø Rein (s4977945)
"""

import torch, os
import numpy as np
import matplotlib.pyplot as plt

from dataset import HipMRIDataset
from modules import ImprovedUNet
from utils import IMAGE_SIZE, NUM_CLASSES, DEVICE

device = DEVICE

# Modified version from the UNet Colab
def show_predictions(model, dataset, title="Segmentation Results", n=3, save_path="./outputs/plots/predict.png"):
    """ 
    Display and save predictions for n samples.

    - model.eval() + no_grad(): deterministic layers, no gradient tracking.
    - For each sample:
        - Build a single-item batch (1, C, H, W) on the right device.
        - Forward pass, then argmax over class channel to get labels (H, W).
        - Plot image/Ground Truth/pred side-by-side (nearest interpolation for masks
          so class boundaries remain crisp).
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            
            # Build a single-sample batch on device
            x = image.unsqueeze(0).to(device).float()
            logits = model(x)

            # Multiclass prediction: argmax over channels → integer labels
            pred_labels = torch.argmax(logits[0], dim=0).cpu().numpy()
            
            # Prepare numpy arrays for plotting
            img_np = image[0].cpu().numpy()
            true_mask_np = true_mask.cpu().numpy()
            
            # Row 1 - Input image
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')
            
            # Row 2 - Ground truth
            axes[1, i].imshow(true_mask_np, interpolation='nearest')
            axes[1, i].set_title(f'True mask {i+1}', fontweight='bold')
            axes[1, i].axis('off')
            
            # Row 3 - Prediction (+ simple pixel accuracy for reference)
            accuracy = np.mean(pred_labels == true_mask_np)
            axes[2, i].imshow(pred_labels, interpolation='nearest')#, vmin=0, vmax=NUM_CLASSES-1)
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.2f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Test dataset, IMPORTANT: transform=True to match training preprocessing
    # Otherwise it get kinda like tv static noisy

    test_ds = HipMRIDataset(img_set='test', size=IMAGE_SIZE, num_classes=NUM_CLASSES, transform=True)

    # Load best checkpoint from train.py
    ckpt = torch.load("./outputs/best.pt", map_location=device)

    model = ImprovedUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt['model_state'])

    # Show + save predictions
    show_predictions(model, test_ds, n=3, save_path="./outputs/plots/predict.png")
