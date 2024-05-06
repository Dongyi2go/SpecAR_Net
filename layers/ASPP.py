import torch

import torch.nn as nn
import torch.nn.functional as F

# Dilated convolution
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# pool -> 1*1 conv -> upsample
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # upsample
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)




class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # Multi-scale dilated convolution
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # pool
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # conv after stack
        # self.project = nn.Sequential(
        #     nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.Dropout(0.5))

        self.project = nn.Sequential(
            nn.Conv2d( out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.95))
        self.l0=nn.Linear(193,192)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.stack(res, dim=-1).mean(-1)
        res=self.project(res)
        res=self.l0(res.mean(2)).permute(0,2,1)

        return res


# if __name__ == '__main__':
#     input=torch.rand(64,512,32,193)
#
#     ap=ASPP(in_channels=512, atrous_rates=[1,3,5], out_channels=512)
#     # print(ap)
#
#     out=ap(input)

    # print(out.shape)