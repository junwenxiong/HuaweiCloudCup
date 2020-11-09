import torch.nn.functional as F
from .unet_parts import *
from unet.SIIS_kernel import SIIS

class UNet_SIIS(nn.Module):
    def __init__(self, 
                n_channels=3,
                n_classes=2,
                siis_size=[32,32],
                width=3,
                kw=9,
                dim=512,
                arch=7,
               ):

        super(UNet_SIIS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.siis = SIIS(siis_size, width , kw, dim, arch)
        
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.siis(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits