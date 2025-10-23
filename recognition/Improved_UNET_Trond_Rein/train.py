"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

import argparse, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from modules import ImprovedUNet, DiceLoss
from dataset import OasisBrainDataset

from utils import IMAGE_SIZE, NUM_CLASSES, EPOCHS, BATCH_SIZE, DEVICE


def loaders(batch_size=BATCH_SIZE):
    train_set = OasisBrainDataset('train', IMAGE_SIZE, NUM_CLASSES)
    val_set = OasisBrainDataset('val', IMAGE_SIZE, NUM_CLASSES)
    test_set = OasisBrainDataset('test', IMAGE_SIZE, NUM_CLASSES)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader


def dice_from_logits(logits, target, num_classes, excl_bg=True):
    prediction = logits.argmax(dim=1)
    target = target.long()
    classes = range(1, num_classes) if excl_bg else range(num_classes)

    dices = []
    for c in classes:
        p = (prediction == c)
        t = (target == c)
        intersection = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        if denom > 0:
            dices.append((2.0 * intersection / denom).item())
    return float(np.mean(dices)) if dices else 1.0

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, device=DEVICE, alpha=0.5):
    model.train()
    total = 0.0 

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)

        ce = ce_loss(logits, masks.long())

        dice = dice_loss(logits, masks)
        loss = alpha * ce + (1 - alpha) * dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * images.size(0)

    return total / len(loader.dataset)


def evaluate(model, loader, ce_loss, dice_loss, num_classes=NUM_CLASSES, device=DEVICE, alpha=0.5):
    model.eval()
    total = 0.0

    dice_list = []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)

        ce = ce_loss(logits, masks.long())

        dice = dice_loss(logits, masks)
        loss = alpha * ce + (1 - alpha) * dice

        total += loss.item() * images.size(0)
        dice_list.append(dice_from_logits(logits, masks, num_classes))

    return total / len(loader.dataset), float(np.mean(dice_list))


def plot_curves(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'],   label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / 'loss_curve.png')
    plt.close()

    # Dice curve
    plt.figure()
    plt.plot(history['val_dice'], label='val dice')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / 'val_dice.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--in_channels', type=int, default=1)   # grayscale by default
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'best.pt'

    # data
    train_loader, val_loader, test_loader = loaders(batch_size=args.batch_size)

    # model
    model = ImprovedUNet(in_channels=args.in_channels, num_classes=NUM_CLASSES, base=args.base, dropout=args.dropout).to(device)

    # losses (simple + stable)
    ce_loss = nn.CrossEntropyLoss()

    dice_loss = DiceLoss(ignore_bg=True)

    # train
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = -1.0
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, alpha=0.5)
        val_loss, val_dice = evaluate(model, val_loader, ce_loss, dice_loss, NUM_CLASSES, device, alpha=0.5)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | dice={val_dice:.4f}")

        # save best by val Dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'model_state': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'in_channels': args.in_channels,
                'base': args.base,
                'dropout': args.dropout,
            }, ckpt_path)
            print(f"  ↳ saved new best to {ckpt_path} (dice {best_dice:.4f})")

    # test with best checkpoint
    print("\nEvaluating on test set with best checkpoint...")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best['model_state'])
    test_loss, test_dice = evaluate(model, test_loader, ce_loss, dice_loss, device, NUM_CLASSES)
    print(f"TEST | loss={test_loss:.4f} | dice={test_dice:.4f}")

    # plots
    plot_curves(history, save_dir)


if __name__ == '__main__':
    main()
