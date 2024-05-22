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

class pix2pix(nn.Module):
    def __init__(self):
        super(pix2pix, self).__init__()
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
            nn.LeakyReLU())

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


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3) #512
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        dx1 = self.deconv1(x8)
        dx2 = self.deconv2(torch.cat((x7, dx1), 1))
        dx3 = self.deconv3(torch.cat((x6, dx2), 1))
        dx4 = self.deconv4(torch.cat((x5, dx3), 1))
        dx5 = self.deconv5(torch.cat((x4, dx4), 1))
        dx6 = self.deconv6(torch.cat((x3, dx5), 1))
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
    d_real_loss = loss_fn(out, torch.ones_like(out))  # Loss of real pictures
    d_real_loss.backward()  # Back propagation
    out = net(hazy, clean)

def Loss2(hazy,synthsized):
    net = PatchGANDiscriminator()
    out = net(hazy,synthsized)
    loss_fn = torch.nn.BCELoss()
    d_fake_loss = loss_fn(out, torch.zeros_like(out))  # Loss of generated pictures
    d_fake_loss.backward()  # Back propagation




if __name__ == '__main__':
    net = pix2pix()
    input_tensor = torch.Tensor(np.random.random((1,1,1024,1024)))
    # net = PatchGANDiscriminator()
    # input_tensor = torch.Tensor(np.random.random((1,1,512,512)))
    # input_tensor1 = torch.Tensor(np.random.random((1, 1, 512, 512)))
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(out.shape)