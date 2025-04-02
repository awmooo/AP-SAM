import torch
import torch.nn as nn

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    生成双线性插值的卷积核权重
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小（假设是正方形，即一个整数表示边长）
    :return: 计算好的卷积核权重张量
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(kernel_size).reshape(-1, 1).float()
    filt = (1 - torch.abs(og - center) / factor) * (1 - torch.abs(og.T - center) / factor)
    # 修正权重张量维度顺序，按照正确的 (in_channels, out_channels, kernel_size, kernel_size)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j] = filt
    return weight
class Bilinear_TransConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride)-> None:
        '''
        生成双线性插值的卷积核权重
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :return:
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans_conv = nn.ConvTranspose2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=0)
    def forward(self,x:torch.Tensor):

        # 初始化权重为双线性插值方式
        weight = bilinear_kernel(self.in_channels, self.out_channels, self.kernel_size)
        self.trans_conv.weight.data.copy_(weight)
        # 初始化偏置为0
        nn.init.constant_(self.trans_conv.bias, 0)
        return self.trans_conv(x)

