import random

import os
from datetime import datetime

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.ImageOps
from PIL import Image
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, root_dir, transform=None, should_invert=False, n=500):

        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
       
        if n > len(self.image_paths):
            self.n = len(self.image_paths)

    def __getitem__(self, index):
        
        img_path = random.choice(self.image_paths)
        img = Image.open(img_path).convert('L')  
        
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)  
        
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
    def __init__(self, root_dir, transform=None, should_invert=False, n=500 , noice_amplitude=1):

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
        img = Image.open(img_path).convert('L')  
        
        
        
        NoiceAdd=AddGaussianNoise(amplitude=self.noice_amplitude)
        
        to_Image=transforms.ToPILImage()
        to_Tensor=transforms.ToTensor()
        
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)  
        
        if self.transform:
            img = self.transform(img)
        
        imgn = NoiceAdd(img=to_Image(img))
        
        return to_Tensor(imgn),img

    def __len__(self):
        return self.n 

class AddMaskPatches(object):
    def __init__(self, mask_size=32, mask_num=1):
        self.mask_size = mask_size  # 掩码块大小
        self.mask_num = mask_num    # 掩码块数量

    def __call__(self, img_tensor):
        
        # 深拷贝原始张量
        masked_img = img_tensor.clone()
        c, h, w = masked_img.shape
        
        for _ in range(self.mask_num):
            # 随机生成掩码块的左上角坐标
            x = random.randint(0, w - self.mask_size)
            y = random.randint(0, h - self.mask_size)
            
            # 将选定的矩形区域置为零
            masked_img[:, y:y+self.mask_size, x:x+self.mask_size] = 0
            
        return masked_img

"""
class DUAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, should_invert=False, n=500, mask_num=1, mask_size=32):
        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        self.mask_num = mask_num  # 掩码块数量
        self.mask_size = mask_size  # 掩码块大小
        self.to_Tensor = transforms.ToTensor()
        # 获取所有图像路径
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
        
        # 颜色反转（如果需要）
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)
        
        # 应用变换（如果有）
        if self.transform:
            img = self.transform(img)
        
        # 转换为Tensor
        
        #img_tensor = self.to_Tensor(img)
        img_tensor=img
        # 添加掩码块
        mask_adder = AddMaskPatches(mask_size=self.mask_size, mask_num=self.mask_num)
        masked_img = mask_adder(img_tensor)
        
        return masked_img, img_tensor

    def __len__(self):
        return self.n
"""

class DarkUnet(nn.Module):
    def __init__(self):
        super().__init__() 
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
            spatial_attention(3),
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
            spatial_attention(5),
            nn.Conv2d(512,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.Tconv_3=nn.Sequential(
            channel_attention(256),
            spatial_attention(9),
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
            spatial_attention(17),
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
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(inplace=True),
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
        y1=self.Tconv_1(torch.cat([x1,y2],1))
        return y1
    
class DUAE_Lable_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, should_invert=False, n=1500 , noice_amplitude=1):
        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        self.noice_amplitude=noice_amplitude
        
        
        self.image_paths = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if os.path.isfile(os.path.join(root_dir, f)) and
            f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
       
        if n > len(self.image_paths):
            self.n = len(self.image_paths)

    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        imgO = Image.open(img_path).convert('L')  
        
        NoiceAdd=AddGaussianNoise(amplitude=self.noice_amplitude)
        to_Image=transforms.ToPILImage()
        to_Tensor=transforms.ToTensor()
        
        if self.should_invert:
            img = Image.eval(imgO, lambda x: 255 - x)  
        
        if self.transform:
            img = self.transform(imgO)
        else:
            img=imgO.copy()
        
        imgn = NoiceAdd(img=to_Image(img))
        
        return to_Tensor(imgn), img, to_Tensor(imgO)

    def __len__(self):
        return self.n
        
def evaluate_dataset(dataloader, net, criterion, device, threshold_U=0,threshold_B=0, save=False, Positive_root='', Unkown_root='',Negative_root='',side_low=0,side_high=1):
    mses = []
    total   = 0
    positive= 0
    unkown  = 0
    negative=0
    net.eval()
    to_Image = torchvision.transforms.ToPILImage()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with torch.no_grad():
        counter = 0
        negative=0
        positive=0
        for inputs, target ,origin in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, target)
            mses.append(loss.item())
            total += 1
            
            if save:
                img_pil = to_Image(origin.squeeze(0).cpu())
                
                counter += 1
                filename = f"{timestamp}_{counter}.png"
                
                if (loss.item() > threshold_U) and (loss.item() < side_high):
                    positive += 1
                    filepath = os.path.join(Positive_root, filename)
                    img_pil.save(filepath)
                elif (loss.item() < threshold_B)and (loss.item()>side_low):
                    negative += 1
                    filepath = os.path.join(Negative_root, filename)
                    img_pil.save(filepath)
                else:
                    unkown +=1
                    filepath = os.path.join(Unkown_root, filename)
                    img_pil.save(filepath)
    
    return mses,total,positive,negative,unkown

def compute_top_percent(maes, percent=0.01):
    sorted_mses = sorted(maes, reverse=True)
    k = max(1, int(len(sorted_mses) * percent))
    return sum(sorted_mses[:k]) / k
 
class DUAE_Dataset_test(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform0=None,transform1=None, should_invert=False, n=500 , noice_amplitude=1):

        self.root_dir = root_dir
        self.transform0 = transform0
        self.transform1 = transform1
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
        img = Image.open(img_path).convert('L')  
        
        
        NoiceAdd=AddGaussianNoise(amplitude=self.noice_amplitude)
        
        to_Image=transforms.ToPILImage()
        to_Tensor=transforms.ToTensor()
        
        if self.should_invert:
            img = Image.eval(img, lambda x: 255 - x)  
        
        if self.transform1:
            imgn = self.transform0(img)
        
        if self.transform0:
            img = self.transform0(img)
        
        
        return imgn,img

    def __len__(self):
        return self.n 
       
def Only_evaluate_dataset(dataloader, net, criterion, device):

    maes = []
    net.eval()
    with torch.no_grad():
        for inputs,target in dataloader:
            inputs,target = inputs.to(device),target.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, target)
            maes.append(loss.item())
    return maes



def mixture_mle(data, max_iter=1000, tol=1e-6,p=0.5):
    
    
    n = len(data)
    mu1, mu2 = np.mean(data) - 0.1, np.mean(data) + 0.1  
    sigma1 = sigma2 = np.std(data)  
    
    log_likelihood_old = -np.inf
    
    for _ in range(max_iter):
        
        comp1 = (1 - p) * norm.pdf(data, mu1, sigma1) + 1e-12
        comp2 = p * norm.pdf(data, mu2, sigma2) + 1e-12
        total = comp1 + comp2 
        gamma = comp2 / total  
        
        n1 = np.sum(1 - gamma)  
        n2 = np.sum(gamma)      
        
        mu1_new = np.sum((1 - gamma) * data) / n1
        mu2_new = np.sum(gamma * data) / n2
        
        sigma1_new = np.sqrt(np.sum((1 - gamma) * (data - mu1_new)**2) / n1)
        sigma2_new = np.sqrt(np.sum(gamma * (data - mu2_new)**2) / n2)
        
        p_new = n2 / n
        
        log_likelihood = np.sum(np.log(comp1 + comp2))
        
        if abs(log_likelihood - log_likelihood_old) < tol:
            break
            
        p, mu1, mu2, sigma1, sigma2 = p_new, mu1_new, mu2_new, sigma1_new, sigma2_new
        log_likelihood_old = log_likelihood
    if mu1>mu2:
        tmp=mu1
        mu1=mu2
        mu2=tmp
        tmp=sigma1
        sigma1=sigma2
        sigma2=tmp
        p=1-p
        
    return p, mu1, sigma1, mu2, sigma2

def plot_and_save_normal_distributions(mean1, std_dev1, mean2, std_dev2,p, filename="normal_distributions.png", dpi=300):
   
    x_min = min(mean1 - 3.5*std_dev1, mean2 - 3.5*std_dev2)
    x_max = max(mean1 + 3.5*std_dev1, mean2 + 3.5*std_dev2)
    x = np.linspace(x_min, x_max, 1000)
    
    pdf1 = norm.pdf(x, mean1, std_dev1)
    pdf2 = norm.pdf(x, mean2, std_dev2)
    
    
    plt.figure(figsize=(10, 6))
    plt.style.use('classic')  
    
    
    plt.plot(x, (1-p)*pdf1, color='blue', linewidth=2, 
             label=f'normal:μ={mean1}, σ={std_dev1}')
    plt.plot(x, p*pdf2, color='red', linewidth=2, 
             label=f'pneumonia:μ={mean2}, σ={std_dev2}')
    plt.plot(x, np.minimum(((1-p)*pdf1)/(p*pdf2),15), color='green', linewidth=2,
             label="normal/pneumonia")
    plt.plot(x, np.minimum((p*pdf2)/((1-p)*pdf1),15), color='yellow', linewidth=2,
             label="pneumonia/normal")
    
    
    plt.title('Normal Distribution Comparison', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()  
    
    print(f"图像已保存为 {filename} (分辨率: {dpi}dpi)")
