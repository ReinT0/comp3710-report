"""
"modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
# Loading all the nessecary libraries //
import torch
import torch.nn as nn

# This is a simple UNet from the colab
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p

        # Encoder (Downsampler)
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        # Bottleneck, the middle part
        self.bottle_neck = DoubleConv(512, 1024)
        
        # Decoder (Upsampler)
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.pool
        # Result
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


    def DoubleConv(self, in_ch, out_ch, dropout_p):
        return nn.Sequentual(
                # First convolution
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLu(negative_slope=0.2, inplace=True),
                nn.Dropout2d(self.dropout_p)
                
                # Second convolution
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLu(negative_slope=0.2, inplace=True),
                nn.Dropout2d(self.dropout_p)
                )

    def DownSampler(self, in_ch, out_ch):
        self.conv = self.DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE)

    def UpSampler(self, in_ch, out_ch):
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.conv = self.DoubleConv(in_ch, out_ch, dropout_p=0.2)
        
