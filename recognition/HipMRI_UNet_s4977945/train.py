"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
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

from modules import ImprovedUNet, DiceLoss
from dataset import HipMRIDataset

from utils import IMAGE_SIZE, NUM_CLASSES, EPOCHS, BATCH_SIZE, DEVICE

def loaders(batch_size=BATCH_SIZE):
    train_set = HipMRIDataset('train', IMAGE_SIZE, NUM_CLASSES, transform=True)
    val_set = HipMRIDataset('val', IMAGE_SIZE, NUM_CLASSES, transform=True)
    test_set = HipMRIDataset('test', IMAGE_SIZE, NUM_CLASSES, transform=True)
    
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, dice_loss, device=DEVICE):
    model.train()
    epoch_loss = []

    for images, masks in tqdm(loader, desc="Training ongoing", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = dice_loss(outputs, masks.long())

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    return np.mean(epoch_loss)


def evaluate(model, loader, dice_loss, device=DEVICE):
    model.eval()
    eval_loss = []
    eval_dice = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validate", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            loss = dice_loss(outputs, masks.long())
            eval_loss.append(loss.item())

        return float(np.mean(eval_loss))

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
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'best.pt'

    # data
    train_loader, val_loader, test_loader = loaders(batch_size=args.batch_size)

    # model
    model = ImprovedUNet(in_channels=args.in_channels, num_classes=NUM_CLASSES, base=args.base, dropout=args.dropout).to(device)

    dice_loss = DiceLoss()

    # train
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = -float("inf")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    val0_loss = evaluate(model, val_loader, dice_loss, device)
    print(f"Epoch 000 | Val loss={val0_loss:.4f}")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, dice_loss, device)
        
        val_loss = evaluate(model, val_loader, dice_loss, device)
        val_dice = 1.0 - val_loss

        print(f"Epoch {epoch:03d} | Train loss={train_loss:.4f} | Val loss={val_loss:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

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
            print(f"  ↳ saved new best to {ckpt_path} (val_dice: {best_dice:.4f})")

    # test with best checkpoint
    print("\nEvaluating on test set with best checkpoint...")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best['model_state'])
    test_loss = evaluate(model, test_loader, dice_loss, device=device)
    test_dice = 1.0 - test_loss
    print(f"TEST | Test loss={test_loss:.4f} | dice={test_dice:.4f}")

    # plots
    plot_curves(history, save_dir)


if __name__ == '__main__':
    main()
