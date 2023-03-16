# 2D U-Net model for semantic segmentation of prostate cancer in MRI images.
# Sources: 
    # - https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
    # - https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A simple convolutional block consisting of two convolutional layers, followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, padding=p)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UNet_Encoder(nn.Module):
    """
    An encoder block, consisting of a convolutional block followed by a max pooling layer.
    The number of filters gets doubled after every block, whereas te dimensions get halved.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class UNet_Decoder(nn.Module):
    """
    A U-Net decoder block, consisting of a transpose convolution for upsampling, and a skip connection from the encoder.
    The upsampled input is concatenated with the encoder output, and then passed through a convolutional block.
    Conversely to the encoder block, the number of filters gets halved after every block, whereas the dimensions are doubled.
    """
    def __init__(self, in_c, out_c, k=2, s=2, p=0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=s, stride=s, padding=p)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet2D(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        # Encoder layers
        self.enc_out_1 = UNet_Encoder(1, 64).to(device)
        self.enc_out_2 = UNet_Encoder(64, 128).to(device)
        self.enc_out_3 = UNet_Encoder(128, 256).to(device)
        self.enc_out_4 = UNet_Encoder(256, 512).to(device)

        self.bottleneck = ConvBlock(512, 1024).to(device) # bottleneck layer

        # Decoder layers
        self.dec_out_1 = UNet_Decoder(1024, 512).to(device)
        self.dec_out_2 = UNet_Decoder(512, 256).to(device)
        self.dec_out_3 = UNet_Decoder(256, 128).to(device)
        self.dec_out_4 = UNet_Decoder(128, 64).to(device)

        self.out_layer = nn.Conv2d(64, 1, kernel_size=1, padding=0).to(device) # output layer

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Encoder forward pass
        skip_1, embs_1 = self.enc_out_1(inputs)
        skip_2, embs_2 = self.enc_out_2(embs_1)
        skip_3, embs_3 = self.enc_out_3(embs_2)
        skip_4, embs_4 = self.enc_out_4(embs_3)
        
        b = self.bottleneck(embs_4) # bottleneck layer

        # Decoder forward pass
        dec_out_1 = self.dec_out_1(b, skip_4)
        dec_out_2 = self.dec_out_2(dec_out_1, skip_3)
        dec_out_3 = self.dec_out_3(dec_out_2, skip_2)
        dec_out_4 = self.dec_out_4(dec_out_3, skip_1)

        seg_mask = self.sigmoid(self.out_layer(dec_out_4)) # segmentation mask output

        return seg_mask