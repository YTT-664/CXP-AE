import random

import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.ImageOps
from PIL import Image

import numpy as np

class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        b, c, h, w = inputs.shape
        max_pool = self.max_pool(inputs).view([b, c])
        avg_pool = self.avg_pool(inputs).view([b, c])
        
        x_maxpool = self.relu(self.fc1(max_pool))
        x_avgpool = self.relu(self.fc1(avg_pool))
        
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)
        
        x = self.sigmoid(x_maxpool + x_avgpool).view([b, c, 1, 1])
        return inputs * x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        x = self.sigmoid(self.conv(x))
        return inputs * x
    
class AE_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, should_invert=True, n=500):

        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        # 若指定数量n大于实际图片数，则用实际数量
        if n > len(self.image_paths):
            self.n = len(self.image_paths)

    def __getitem__(self, index):
        # 随机选择一张图片
        img_path = random.choice(self.image_paths)
        img = Image.open(img_path).convert('L')  # 转为灰度图
        
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)  # 颜色反转
        
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.n 

class DarkNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkNetBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2),
            nn.SiLU(),
            nn.Conv2d(in_channels//2,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            )
    def forward(self,x):
        return x+self.block(x)
    
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            DarkNetBlock(64),
            
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            DarkNetBlock(128),
            DarkNetBlock(128),
            channel_attention(128),
            spatial_attention(13),
            
            nn.Conv2d(128,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            DarkNetBlock(256),
            channel_attention(256),
            spatial_attention(7),
            
            
            nn.Conv2d(256,512,3,2,1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            DarkNetBlock(512),
            channel_attention(512),
            spatial_attention(3),
            
            nn.Conv2d(512,1024,3,2,1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            
        )
    def forward(self,x):
        return self.net(x)
        
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=1024, output_channels=1):
        super().__init__()
        # 转置卷积块（逐步上采样）
        self.decoder = nn.Sequential(
            # 输入: [batch, 1024, 13, 13]
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出: [batch, 512, 26, 26]
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输出: [batch, 256, 52, 52]
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输出: [batch, 128, 104, 104]
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DarkNetBlock(64),
            DarkNetBlock(64),
            channel_attention(64),
            # 输出: [batch, 64, 208, 208]
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DarkNetBlock(32),
            DarkNetBlock(32),
            DarkNetBlock(32),
            DarkNetBlock(32),
            channel_attention(32),
            # 输出: [batch, 64, 416, 416]
            
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Tanh()  # 输出像素值归一化到[-1, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.net_E=DarkNet53()
        self.net_D=ImageDecoder()
    def forward(self,x):
        
        x=self.net_E(x)
        y=x
        y=y.squeeze()
        x=self.net_D(x)
        return x,y
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img_array = np.array(img)  # 转换为NumPy数组
        # 适配灰度图像：如果是二维，扩展为三维（h, w, 1）
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]  # 增加通道维度
        
        h, w, c = img_array.shape  # 此时c=1（灰度）
        # 生成噪声（三维数组）
        N = self.amplitude * np.random.normal(
            loc=self.mean, scale=self.variance, size=(h, w, 1)
        )
        # 扩展噪声到与图像相同的通道数
        N = np.repeat(N, c, axis=2)
        img_array = img_array + N
        img_array = np.clip(img_array, 0, 255)  # 裁剪到[0,255]
        img_array = img_array.astype('uint8')
        
        # 转换回PIL图像
        if c == 1:
            img_array = img_array.squeeze(axis=2)  # 移除通道维度（变回二维）
        return Image.fromarray(img_array).convert('L')  # 确保输出灰度图

