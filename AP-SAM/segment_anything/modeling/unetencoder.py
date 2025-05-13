
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
        # self.doubledown = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.down3 = (Down(256, 512))
        # self.down4 = (Down(512, 768))
        # self.out =   DoubleConv(768, 256)






    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        # x3 = self.doubledown(x2)
        # x = self.down3(x2)
        # x = self.down4(x)
        # x = self.out(x)



        return x,x1,x2

