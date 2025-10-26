import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NUM_CLASSES


# Helper for the UNet
class PreActivationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()

        # First convolution
        self.norm1 = nn.InstanceNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.act = nn.SiLU()
        
        self.drop = nn.Dropout2d(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
                
    def forward(self, x):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.drop(y)

        y = self.conv2(self.act(self.norm2(y)))

        return y + self.skip(x)


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
        
        # Bottleneck (Middle part)
        self.bottleneck = PreActivationBlock(f3, f3, dropout)
        
        # Decoder (Upsampler)
        self.up3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)        
        self.decode3 = PreActivationBlock(f2+f2, f2, dropout)

        self.up2 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)        
        self.decode2 = PreActivationBlock(f1+f1, f1, dropout)

        self.up1 = nn.ConvTranspose2d(f1, f0, kernel_size=2, stride=2)        
        self.decode1 = PreActivationBlock(f0+f0, f0, dropout)

        self.segment3 = nn.Conv2d(f2, num_classes, kernel_size=1)
        self.segment2 = nn.Conv2d(f1, num_classes, kernel_size=1)
        self.segment1 = nn.Conv2d(f0, num_classes, kernel_size=1)
        

    def forward(self, x):
        H, W = x.shape[-2:]

        e1 = self.encode1(x)
        e2 = self.encode2(self.down1(e1))
        e3 = self.encode3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.decode3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.decode2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decode1(torch.cat([self.up1(d2), e1], dim=1))

        out1 = self.segment1(d1)
        out2 = F.interpolate(self.segment2(d2), size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.segment3(d3), size=(H, W), mode='bilinear', align_corners=False)
        
        out = out1 + out2 + out3
        
        return out


class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, targets):
        if predictions.ndim == 4:
            predictions = predictions.squeeze(1)

        predictions = F.softmax(predictions, dim=1)

        target_oh = F.one_hot(targets.long(), self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = (predictions * target_oh).sum(dims)
        denom = (predictions + target_oh).sum(dims)

        dice_coeff = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        loss = 1.0 - dice_coeff.mean()

        return loss
