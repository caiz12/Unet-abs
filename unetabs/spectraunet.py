""" Module for the Spectra Unet class"""
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels, ksize=3, padding=(0,1)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (1, ksize), padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, (1, ksize), padding=padding),
        nn.ReLU(inplace=True)
    )


class SpectraUNet(nn.Module):

    def __init__(self):
        super(SpectraUNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        #
        self.maxpool = nn.MaxPool2d((1, 2))
        #
        self.upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True)

        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(16 + 32, 16)

        self.conv_last = nn.Conv2d(16, 1, 1)  #

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        # Come back up

        x = self.upsample(x)
        # import pdb; pdb.set_trace()

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out