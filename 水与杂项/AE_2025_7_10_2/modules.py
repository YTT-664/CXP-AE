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
            #DarkNetBlock(64),
            #DarkNetBlock(64),
            channel_attention(64),
            # 输出: [batch, 64, 208, 208]
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #DarkNetBlock(32),
            #DarkNetBlock(32),
            #DarkNetBlock(32),
            #DarkNetBlock(32),
            channel_attention(32),
            # 输出: [batch, 64, 416, 416]
            
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Tanh()  
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
        img_array = np.array(img)  
        
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]  
        
        h, w, c = img_array.shape  

        N = self.amplitude * np.random.normal(
            loc=self.mean, scale=self.variance, size=(h, w, 1)
        )
        
        N = np.repeat(N, c, axis=2)
        img_array = img_array + N
        img_array = np.clip(img_array, 0, 255)  
        img_array = img_array.astype('uint8')
        
        if c == 1:
            img_array = img_array.squeeze(axis=2)
        return Image.fromarray(img_array).convert('L') 
    
class DUAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, should_invert=True, n=500 , noice_amplitude=1):

        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        self.noice_amplitude=noice_amplitude
        
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        if n > len(self.image_paths):
            self.n = len(self.image_paths)

    def __getitem__(self, index):
        img_path = random.choice(self.image_paths)
        img = Image.open(img_path).convert('L')  # 转为灰度图
        
        
        
        NoiceAdd=AddGaussianNoise(amplitude=self.noice_amplitude)
        
        imgn = NoiceAdd(img=img)
        
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)  # 颜色反转
        
        if self.transform:
            img = self.transform(img)
            imgn = self.transform(imgn)
        
        return imgn,img

    def __len__(self):
        return self.n 
    
class DarkUnet(nn.Module):
    def __init__(self):
        self.conv_1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            DarkNetBlock(64)
        )
        self.conv_3=nn.Sequential(
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            DarkNetBlock(128),
            DarkNetBlock(128),
            channel_attention(128),
            spatial_attention(13)
        )
        self.conv_4=nn.Sequential(
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
            spatial_attention(7)
        )
        self.conv_5=nn.Sequential(
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
            spatial_attention(3)
        )
        self.conv_6=nn.Sequential(
            nn.Conv2d(512,1024,3,2,1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            DarkNetBlock(1024),
            channel_attention(1024)
        )
        self.Tconv_6=nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.Tconv_5=nn.Sequential(
            channel_attention(1024),
            nn.Conv2d(1024,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
        )
        self.Tconv_4=nn.Sequential(
            channel_attention(512),
            nn.Conv2d(512,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.Tconv_3=nn.Sequential(
            channel_attention(256),
            nn.Conv2d(256,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.Tconv_2=nn.Sequential(
            channel_attention(128),
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.Tconv_1=nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.Conv2d(32,32,3,1,1),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh()  
        )
    def forward(self,x):
        x1=self.conv_1(x)
        x2=self.conv_2(x1)
        x3=self.conv_3(x2)
        x4=self.conv_4(x3)
        x5=self.conv_5(x4)
        x6=self.conv_6(x5)
        y6=self.Tconv_6(x6)
        y5=self.Tconv_5(torch.cat([x5,y6],1))
        y4=self.Tconv_4(torch.cat([x4,y5],1))
        y3=self.Tconv_3(torch.cat([x3,y4],1))
        y2=self.Tconv_2(torch.cat([x2,y3],1))
        y1=self.Tconv_2(torch.cat([x1,y2],1))
        return y1
        
        