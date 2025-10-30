"""
This file is for the different components of the model.
ImprovedUNet and a DiceLoss to calculate loss between epochs.
Author: Trond Jakob Grø Rein (s4977945)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NUM_CLASSES

# Helper for the UNet
class PreActivationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()

        # Pre-activation for first convolution layer
        self.norm1 = nn.InstanceNorm2d(in_ch) # Instance norm is better (more stable) for smaller batches
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Pre-activation for the second convolution layer
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Activation layer. Both LeakyReLU and SiLu works 
        #self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.act = nn.SiLU() # SiLU does have a little smoother in the break than LeakyReLU
        
        self.drop = nn.Dropout2d(dropout) # Dropout, help with overfitting
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity() # 1x1 skip to next step
                
    def forward(self, x):
        
        # THe pre-activation: normalization -> activation -> convolution
        y = self.conv1(self.act(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.act(self.norm2(y)))

        return y + self.skip(x) # adds the last step, the skip in the UNet that makes it look like a U.


# The ImprovedUNet, I first took inspiration from the UNet from the colab used in class, then expanded it
# to work as the Improved UNet showed in the slides
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, base=32, dropout=0.1):
        super().__init__()
        f0, f1, f2, f3 = base, base*2, base*4, base*8

        # Encoder (Downsampler)
        self.encode1 = PreActivationBlock(in_channels, f0, dropout)
        self.down1 = nn.Conv2d(f0, f1, kernel_size=3, stride=2, padding=1)

        self.encode2 = PreActivationBlock(f1, f1, dropout)
        self.down2 = nn.Conv2d(f1, f2, kernel_size=3, stride=2, padding=1)

        self.encode3 = PreActivationBlock(f2, f2, dropout)
        self.down3 = nn.Conv2d(f2, f3, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck (Middle part) the deepest features
        self.bottleneck = PreActivationBlock(f3, f3, dropout)
        
        # Decoder (Upsampler)
        self.up3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)        
        self.decode3 = PreActivationBlock(f2+f2, f2, dropout)

        self.up2 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)        
        self.decode2 = PreActivationBlock(f1+f1, f1, dropout)

        self.up1 = nn.ConvTranspose2d(f1, f0, kernel_size=2, stride=2)        
        self.decode1 = PreActivationBlock(f0+f0, f0, dropout)
        
        # Deep supervision heads: predicts on multiple scales
        self.segment3 = nn.Conv2d(f2, num_classes, kernel_size=1) # Deepest decoder
        self.segment2 = nn.Conv2d(f1, num_classes, kernel_size=1) # Mid-level
        self.segment1 = nn.Conv2d(f0, num_classes, kernel_size=1) # Final decoder
        

    def forward(self, x):
        H, W = x.shape[-2:]
        
        # Encoder
        e1 = self.encode1(x)
        e2 = self.encode2(self.down1(e1))
        e3 = self.encode3(self.down2(e2))

        # Bottleneck
        b = self.bottleneck(self.down3(e3))
        
        # Decoder, with skip connections
        d3 = self.decode3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.decode2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decode1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Deep supervision, creates segment maps at three different depth, and upsamples to make them all full resolution
        out1 = self.segment1(d1)
        out2 = F.interpolate(self.segment2(d2), size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.segment3(d3), size=(H, W), mode='bilinear', align_corners=False)
        
        # Combining them makes use of earlier decoder layer to make useful predictions, help with faster convergence
        out = out1 + out2 + out3
        
        return out

# NOT IN USE NOW, MOVED TO DicePlusCE
# This is a modified version from the UNet colab
class DiceLoss(nn.Module):
    # Description copied from the Colab
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """

    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # Passes in logits (N, C, H, W) not target (N, -, H, W) like in the Colab
        predictions = F.softmax(logits, dim=1) # Since Logit, do softmax here.

        # One-hot encode to match the class
        target_oh = F.one_hot(targets.long(), self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = (predictions * target_oh).sum(dims) # |X ∩ Y|
        denom = (predictions + target_oh).sum(dims) # |X| + |Y|

        # (2 * |X ∩ Y|) / (|X| + |Y| + smooth), smooth is so no divide by 0
        dice_coeff = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        loss = 1.0 - dice_coeff[1:].mean() # ignores background

        return loss


class DicePlusCELoss(nn.Module):
    """ This is a answer from ChatGPT why use Dice with Cross-entropy over just Dice
    Combined loss = Dice loss + λ * Cross-Entropy.

    Why combine?
      • Dice term: optimizes overlap and handles imbalance.
      • CE term: stabilizes per-class learning and sharpens boundaries.
      • Total loss is unbounded (>1 is normal) because CE is unbounded.

    Notes
      • Keep logits (no softmax) for CE; apply softmax for Dice.
      • Background (class 0) is ignored in the Dice term (common practice).
      • Tune ce_weight (λ): 0.2–0.5 usually works well.
    """
    # Otherwise from the ce, this is the same as DiceLoss over
    def __init__(self, ce_weight=0.3, num_classes=NUM_CLASSES, smooth=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        
        ce = F.cross_entropy(logits, targets)
        predictions = F.softmax(logits, dim=1)
        target_oh = F.one_hot(targets.long(), self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = (predictions * target_oh).sum(dims)
        denom = (predictions + target_oh).sum(dims)

        dice_coeff = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_coeff = dice_coeff[1:].mean() # ignores bckground

        loss = 1.0 - dice_coeff
        loss_with_ce = loss + self.ce_weight * ce
        
        return loss_with_ce
