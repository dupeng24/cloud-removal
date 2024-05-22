# coding:utf-8
import time

import torch.nn as nn
import torchvision
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import os
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        channel_attention = torch.sigmoid(avg_out + max_out)

        return channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, in_channels,kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, kernel_size=x.size(2))
        max_pool, _ = F.max_pool2d(x, kernel_size=x.size(2), return_indices=True)
        spatial_attention = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))

        return spatial_attention

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(in_channels,kernel_size)

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x

class pix2pixCBAM(nn.Module):
    def __init__(self):
        super(pix2pixCBAM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())

        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.deconv8 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, padding=1,stride=2),
            nn.Tanh())

        self.CBAM1 = CBAM(512)
        self.CBAM2 = CBAM(256)
        self.CBAM3 = CBAM(128)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3) #512
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x8 = self.CBAM1(x8)
        dx1 = self.deconv1(x8)
        dx1 =  self.CBAM1(dx1)
        dx2 = self.deconv2(torch.cat((x7, dx1), 1))
        dx2 = self.CBAM1(dx2)
        dx3 = self.deconv3(torch.cat((x6, dx2), 1))
        dx3 = self.CBAM1(dx3)
        dx4 = self.deconv4(torch.cat((x5, dx3), 1))
        dx4 = self.CBAM1(dx4)
        dx5 = self.deconv5(torch.cat((x4, dx4), 1))
        dx5 = self.CBAM2(dx5)
        dx6 = self.deconv6(torch.cat((x3, dx5), 1))
        dx6 = self.CBAM3(dx6)
        dx7 = self.deconv7(torch.cat((x2, dx6), 1))
        result = self.deconv8(torch.cat((x1, dx7), 1))

        return result



class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, padding=1,stride=2),
            nn.LeakyReLU())
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

    def forward(self, anno, img):
        x = torch.cat([anno, img], axis=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x


def Loss1(hazy,clean):
    net = PatchGANDiscriminator()
    out = net(hazy,clean)
    loss_fn = torch.nn.BCELoss()
    d_real_loss = loss_fn(out, torch.ones_like(out))  # # Loss of real pictures
    d_real_loss.backward()  # Back propagation
    out = net(hazy, clean)

def Loss2(hazy,synthsized):
    net = PatchGANDiscriminator()
    out = net(hazy,synthsized)
    loss_fn = torch.nn.BCELoss()
    d_fake_loss = loss_fn(out, torch.zeros_like(out))  # Loss of generated pictures
    d_fake_loss.backward()  # Back propagation




if __name__ == '__main__':
    net = pix2pixCBAM()
    input_tensor = torch.Tensor(np.random.random((1,1,1024,1024)))
    # net = PatchGANDiscriminator()
    # input_tensor = torch.Tensor(np.random.random((1,1,512,512)))
    # input_tensor1 = torch.Tensor(np.random.random((1, 1, 512, 512)))
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(out.shape)