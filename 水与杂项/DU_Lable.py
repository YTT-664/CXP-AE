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
import matplotlib.pyplot as plt  
import os  
import PIL.ImageOps
from PIL import Image
from PIL import ImageFile
from datetime import datetime

def evaluate_dataset(dataloader, net, criterion, device, threshold_U=0,threshold_B=0, save=False, Positive_root='', Unkown_root='',Negative_root=''):
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
                
                if loss.item() > threshold_U:
                    positive += 1
                    filepath = os.path.join(Positive_root, filename)
                    img_pil.save(filepath)
                if loss.item() < threshold_B:
                    negative += 1
                    filepath = os.path.join(Negative_root, filename)
                    img_pil.save(filepath)
    
    return mses,total,positive,negative,unkown

def compute_top_percent(maes, percent=0.01):
    sorted_mses = sorted(maes, reverse=True)
    k = max(1, int(len(sorted_mses) * percent))
    return sum(sorted_mses[:k]) / k

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


def main():
    
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("reconstruction_comparisons", exist_ok=True)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=51),
        transforms.Resize([416,416]),
        transforms.ToTensor(),
    ])
    
    Test_Dataset = modules.DUAE_Dataset(
        root_dir="archive/chest_xray/chest_xray/train/NORMAL",
        transform=transform,
        should_invert=False,
        n=500,
        noice_amplitude=26
    )
    Test_dataloader = DataLoader(
        Test_Dataset, 
        shuffle=True, 
        batch_size=1,
        pin_memory=True
    )
    
    Normal_Dataset = modules.DUAE_Lable_Dataset(
        root_dir="archive/chest_xray/chest_xray/train/NORMAL",
        transform=transform,
        should_invert=False,
        n=100,
        noice_amplitude=26
    )
    Normal_dataloader = DataLoader(
        Normal_Dataset, 
        shuffle=True, 
        batch_size=1,
        
        pin_memory=True
    )
    Pneumonia_Dataset = modules.DUAE_Lable_Dataset(
        root_dir="archive/chest_xray/chest_xray/train/PNEUMONIA",
        transform=transform,
        should_invert=False,
        n=100,
        noice_amplitude=26
    )
    Pneumonia_dataloader = DataLoader(
        Pneumonia_Dataset, 
        shuffle=True, 
        batch_size=1,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modules.DarkUnet().to(device)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    net.load_state_dict(torch.load('pth/model_final.pth'))
    net.eval()
    test=Only_evaluate_dataset(Test_dataloader,net,criterion,device)
    test_mean = sum(test) / len(test)
    test_var = sum((x - test_mean)**2 for x in test) / len(test)
    THRESHOLD_up=test_mean+2*(test_var**0.5)
    THRESHOLD_bottom=test_mean
    data1,total1,positive1,negative1,_=evaluate_dataset(Normal_dataloader,net,criterion,device,THRESHOLD_up,THRESHOLD_bottom,False,'Pseudo_Data/abnormal','Pseudo_Data/unknown','Pseudo_Data/normal')
    data2,total2,positive2,negative2,_=evaluate_dataset(Pneumonia_dataloader,net,criterion,device,THRESHOLD_up,THRESHOLD_bottom,False,'Pseudo_Data/abnormal','Pseudo_Data/unknown','Pseudo_Data/normal')
    #print("TP:",positive2)
    #print("TN:",negative1)
    #print("FP:",positive1)
    #print("FN:",negative2)
    data=data1+data2
    p, mu1, sigma1, mu2, sigma2=modules.mixture_mle(data)
    plt.figure(figsize=(10, 6))
    plt.hist(
        [data1, data2],
        bins=20,
        alpha=0.8,
        label=['NORMAL', 'PNEUMONIA'],
        color=['green', 'red'],
        stacked=False
    )
    #plt.axvline(x=mu1, color='blue', linestyle='--', label=f'Threshold ({mu1})')
    #plt.axvline(x=mu2, color='red', linestyle='--', label=f'Threshold ({mu2})')
    #plt.axvline(x=mu1+2*sigma1, color='blue', linestyle='--', label=f'Threshold ({mu1+2*sigma1})')
    #plt.axvline(x=mu2-2*sigma2, color='red', linestyle='--', label=f'Threshold ({mu2-2*sigma2})')
    plt.xlabel('L1 Loss (MAE)')
    plt.ylabel('Frequency')
    plt.title('L1 Loss Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_distribution.pdf')
    plt.close()
    print("Loss distribution plot saved as loss_distribution.png")

if __name__ == "__main__":
    #os.makedirs('Pseudo_Data/abnormal', exist_ok=True)
    #os.makedirs('Pseudo_Data/unkown', exist_ok=True)
    #os.makedirs('Pseudo_Data/normal', exist_ok=True)
    main()