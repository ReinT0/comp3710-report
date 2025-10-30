"""
train.py, Have code for training, validating, testing and saving the model. Also plots metrics
Author: Trond Jakob Grø Rein (s4977945)
"""

import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from modules import ImprovedUNet, DicePlusCELoss
from dataset import HipMRIDataset

from utils import IMAGE_SIZE, NUM_CLASSES, EPOCHS, BATCH_SIZE, DEVICE

def loaders(batch_size=BATCH_SIZE):
    # Build the datasets
    train_set = HipMRIDataset('train', IMAGE_SIZE, NUM_CLASSES, transform=True)
    val_set = HipMRIDataset('val', IMAGE_SIZE, NUM_CLASSES, transform=True)
    test_set = HipMRIDataset('test', IMAGE_SIZE, NUM_CLASSES, transform=True)
    
    pin = torch.cuda.is_available() # Boosts GPU usage if possible
    # Load the data with DataLoader from torch, only shuffle train set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, dice_loss, device=DEVICE):
    """
    One training epoch:
      - zero_grad -> forward -> loss -> backward -> optimizer.step
      - accumulate mean loss over all batches
    """
    model.train()
    epoch_loss = []

    for images, masks in tqdm(loader, desc="Training ongoing", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images) # Logits (N, C, H, W)

        loss = dice_loss(outputs, masks.long()) # DicePlusCELoss (and DiceLoss) expects logits and masks in ints

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    return float(np.mean(epoch_loss))

def dice_per_class(model, loader, num_classes=NUM_CLASSES, device=DEVICE):
    """ 
    True Dice metric per class
    Steps:
      - eval() + no_grad()
      - logits -> argmax to labels
      - accumulate intersection and (pred + target) per class
    """
    model.eval() # Put the model in evaluation mode
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) # Logits (N, C, H, W)
            pred = torch.argmax(logits, dim=1) # Target (N, H, W)
            for c in range(num_classes):
                p = (pred == c).float()
                t = (y == c).float()
                inter[c] += (p*t).sum()
                denom[c] += (p+t).sum()
    dice = (2*inter) / (denom.clamp_min(1e-6)) # the clamp_min is the same as smooth, no divide by 0
    return dice # List


def evaluate(model, loader, dice_loss, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Validation pass that returns:
      - mean loss over the loader
      - TRUE mean Dice excluding background (computed from argmax labels)
    """
    
    model.eval()
    eval_loss = []
    
    intersection = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validate", leave=False):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = dice_loss(logits, masks.long())
            eval_loss.append(loss.item())
            
            prediction = torch.argmax(logits, dim=1)
            for c in range(num_classes):
                p = (prediction == c).float()
                t = (masks == c).float()
                intersection[c] += (p * t).sum()
                denom[c] += (p + t).sum()

        val_loss = float(np.mean(eval_loss))
        dices = (2 * intersection) / denom.clamp_min(1e-6)
        val_dice_true = dices[1:].mean().item()  # exclude background

        return val_loss, val_dice_true

def plot_curves(history, out_dir):
    """
    Save training curves to out_dir/plots/:
      - loss_curve.png   - train vs val loss
      - val_dice.png     - TRUE validation Dice over epochs
    """
    out_dir = Path(out_dir)
    (out_dir / 'plots').mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'],   label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / 'plots/loss_curve.png')
    plt.close()

    # Dice curve
    plt.figure()
    plt.plot(history['val_dice'], label='val dice')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / 'plots/val_dice.png')
    plt.close()


def main():
    # --------------------
    # CLI + configuration
    # --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',         type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',     type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--save_dir',       type=str,   default='./outputs')
    parser.add_argument('--in_channels',    type=int,   default=1)
    parser.add_argument('--base',           type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Output folder and checkpoint path
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'best.pt'

    # data
    train_loader, val_loader, test_loader = loaders(batch_size=args.batch_size)

    # model
    model = ImprovedUNet(in_channels=args.in_channels, num_classes=NUM_CLASSES, base=args.base, dropout=args.dropout).to(device)
    
    # Dice + CE: CE stabilizes per-class logits, Dice optimizes overlap
    dice_loss = DicePlusCELoss(ce_weight=0.3, num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # train
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = -float("inf")
    
    # Baseline validation before any training
    val0_loss, val0_dice = evaluate(model, val_loader, dice_loss, device=device)
    print(f"Epoch 000 | Val loss={val0_loss:.4f} | Val Dice={val0_dice:.4f}")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, dice_loss, device)
        val_loss, val_dice= evaluate(model, val_loader, dice_loss, device=device)

        print(f"Epoch {epoch:03d} | Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | Val Dice={val_dice:.4f}")

        # Adding data for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        # Save the best model by TRUE validation Dice (excl. background)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'model_state': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'in_channels': args.in_channels,
                'base': args.base,
                'dropout': args.dropout,
            }, ckpt_path)
            print(f"  ↳ saved new best to {ckpt_path} (val_dice: {best_dice:.4f})")

    # test with best checkpoint
    print("\nEvaluating on test set with best checkpoint...")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best['model_state'])

    test_dice = dice_per_class(model, test_loader, device=device)
    mean_dice_no_bg = test_dice[1:].mean().item()

    print("TEST | Per-class Dice:", [float(x) for x in test_dice.tolist()])
    print(f"TEST | Mean Dice (no background): {mean_dice_no_bg:.4f}")

    # plots
    plot_curves(history, save_dir)


if __name__ == '__main__':
    main()
