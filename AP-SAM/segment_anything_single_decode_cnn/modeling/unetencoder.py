
""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .common import LayerNorm2d

class UNetEncoder(nn.Module):
    def __init__(self, n_channels):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels


        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.out =   DoubleConv(1024, 256)






    def forward(self, x):
        x = self.inc(x)
        x3 = self.down1(x)
        x2 = self.down2(x3)
        x = self.down3(x2)
        x = self.down4(x)
        x = self.out(x)



        return x,x2,x3

