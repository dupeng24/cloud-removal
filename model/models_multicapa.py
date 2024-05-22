# coding:utf-8
import time

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import os
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


# encoder
######################
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))  #下整数
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False) #1*1调整通道数
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1) #width=26
        for i in range(self.nums): #nums = 3
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            a = self.pool(spx[self.nums])
            out = torch.cat((out, a), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):  #layers=[3,4,23,3]
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # block,64,3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv0(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x3 = self.layer3(x_layer2)  # x16

        return x3, x_layer1, x_layer2,x




# decoder
######################

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa0 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.PReLU(channel)
        )
        self.conv0 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(channel // 4,channel // 4, kernel_size=3, stride=1, padding=3, dilation=3, bias=True)
        self.conv2 = nn.Conv2d(channel // 4,channel // 4, kernel_size=3, stride=1, padding=5, dilation=5, bias=True)
        self.conv3 = nn.Conv2d(channel // 4, channel // 4, kernel_size=3, stride=1, padding=7, dilation=7, bias=True)
        self.conv4 = nn.Conv2d(channel // 4, channel // 4, kernel_size=3, stride=1, padding=9, dilation=9, bias=True)
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.PReLU(channel // 8),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.pa0(x)
        x1 = self.conv0(x0)
        x2 = self.conv1(x1)
        x3 = self.conv2(x1)
        x4 = self.conv3(x1)
        x5 = self.conv4(x1)
        x6 = torch.cat([x2, x3, x4,x5], dim=1)
        y = self.pa(x6)
        return x * y

class ResnetGlobalAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ResnetGlobalAttention, self).__init__()

        self.feature_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        zx = y.squeeze(-1)
        zy = zx.permute(0, 2, 1)
        zg = torch.matmul(zy, zx)

        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
        v = v.unsqueeze_(-1).permute(1, 2, 0)

        atten = self.conv(y.squeeze(-1).transpose(-1, -2))
        atten = atten + v
        atten = self.conv_end(atten)
        atten = atten.permute(0,2,1).unsqueeze(-1)

        atten_score = self.soft(atten)

        return x * atten_score

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.ca0 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1,padding=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.PReLU(channel)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.PReLU(channel // 8),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.ca0(x)
        y = self.avg_pool(x0)
        y = self.ca(y)
        return x * y


class DecloudBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DecloudBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        #self.calayer = CALayer(dim)
        self.calayer = ResnetGlobalAttention(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class Decloud(nn.Module):
    def __init__(self, imagenet_model):
        super(Decloud, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4)
        self.mid_conv = DecloudBlock(default_conv, 1024, 3)
        self.up_block1 = nn.PixelShuffle(2)
        self.attention1 = DecloudBlock(default_conv, 256, 3)
        self.attention2 = DecloudBlock(default_conv, 192, 3)
        self.attention3 = DecloudBlock(default_conv, 128, 3)
        self.enhancer = Enhancer(32, 32)

    def forward(self, input):
        x, x_layer1, x_layer2, x3 = self.encoder(input)  # x3是第一层

        x_mid = self.mid_conv(x)

        x = self.up_block1(x_mid)
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)
        x = self.up_block1(x)
        x = self.attention2(x)

        x = torch.cat((x, x_layer1, x3), 1)
        x = self.up_block1(x)
        x = self.attention3(x)
        x = self.up_block1(x)

        dout2 = self.enhancer(x)

        return dout2


class cloudCPA(nn.Module):
    def __init__(self, imagenet_model,model):
        super(cloudCPA, self).__init__()

        self.feature_extract = Decloud(imagenet_model)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 1, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):

        feature = self.feature_extract(input)

        clean = self.tail(feature)

        return clean


class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))

if __name__ == '__main__':
    net = cloudCPA(None,None)
    input_tensor = torch.Tensor(np.random.random((1,1,1024,1024)))
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(input_tensor.shape)
    print(out.shape)