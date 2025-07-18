import random
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.ImageOps
from PIL import Image
from torch.utils.data import DataLoader
import modules
import time
import matplotlib.pyplot as plt  # 添加matplotlib库
import os  # 添加os库用于目录操作
import PIL.ImageOps
from PIL import Image
from PIL import ImageFile

def compute_top_percent(mses, percent=0.01):
    """计算最大的percent% MSE值的平均值"""
    sorted_mses = sorted(mses, reverse=True)
    k = max(1, int(len(sorted_mses) * percent))
    return sum(sorted_mses[:k]) / k

def compute_bottom_percent(mses, percent=0.01):
    """计算最小的percent% MSE值的平均值"""
    sorted_mses = sorted(mses)  # 升序排列
    k = max(1, int(len(sorted_mses) * percent))
    return sum(sorted_mses[:k]) / k

def evaluate_dataset(dataloader, net, criterion, device):
    """评估数据集并返回MSE列表"""
    mses = []
    net.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, inputs)
            mses.append(loss.item())
    return mses

def main():
    
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("reconstruction_comparisons", exist_ok=True)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=0),
        transforms.Resize([160,160]),
        #transforms.RandomCrop([832, 832]),
        transforms.ToTensor(),
    ])
    Normal_Dataset = modules.AE_Dataset(
        root_dir="archive/chest_xray/chest_xray/test/NORMAL",
        transform=transform,
        should_invert=False,
        n=200
    )
    Normal_dataloader = DataLoader(
        Normal_Dataset, 
        shuffle=True, 
        batch_size=1
    )
    Pneumonia_Dataset = modules.AE_Dataset(
        root_dir="archive/chest_xray/chest_xray/test/PNEUMONIA",
        transform=transform,
        should_invert=False,
        n=200
    )
    Pneumonia_dataloader = DataLoader(
        Pneumonia_Dataset, 
        shuffle=True, 
        batch_size=1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modules.AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    net.load_state_dict(torch.load('pth/model_final.pth'))
    net.eval()
    
    normal_mses = evaluate_dataset(Normal_dataloader, net, criterion, device)
    pneumonia_mses = evaluate_dataset(Pneumonia_dataloader, net, criterion, device)
    
    normal_mean = sum(normal_mses) / len(normal_mses)
    normal_var = sum((x - normal_mean)**2 for x in normal_mses) / len(normal_mses)
    normal_top = compute_top_percent(normal_mses,percent=0.1)

    pneumonia_mean = sum(pneumonia_mses) / len(pneumonia_mses)
    pneumonia_var = sum((x - pneumonia_mean) **2 for x in pneumonia_mses) / len(pneumonia_mses)
    pneumonia_top = compute_top_percent(pneumonia_mses)

    normal_bottom = compute_bottom_percent(normal_mses)
    pneumonia_bottom = compute_bottom_percent(pneumonia_mses,percent=0.1)

    # 打印结果
    print("\n" + "="*50)
    print(f"Normal Dataset MSE Statistics:")
    print(f"- Mean: {normal_mean:.6f}")
    print(f"- Variance: {normal_var:.6f}")
    print(f"- Mean of top 10% MSE: {normal_top:.6f}")
    print(f"- Mean of bottom 1% MSE: {normal_bottom:.6f}")

    print("\n" + "="*50)
    print(f"Pneumonia Dataset MSE Statistics:")
    print(f"- Mean: {pneumonia_mean:.6f}")
    print(f"- Variance: {pneumonia_var:.6f}")
    print(f"- Mean of top 1% MSE: {pneumonia_top:.6f}")
    print(f"- Mean of bottom 10% MSE: {pneumonia_bottom:.6f}")
    print("="*50)
    
    
    
if __name__ == "__main__":
    main()
   