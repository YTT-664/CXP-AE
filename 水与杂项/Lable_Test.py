import random
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.ImageOps
from PIL import Image
from torch.utils.data import DataLoader
import AE_2025_7_9_3.modules as modules
import time
import matplotlib.pyplot as plt  
import os  
import PIL.ImageOps
from PIL import Image
from PIL import ImageFile

def main():
    
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("reconstruction_comparisons", exist_ok=True)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=15),
        transforms.Resize([1056,1056]),
        #transforms.RandomCrop([832, 832]),
        transforms.ToTensor(),
    ])
    
    Unkown_Dataset = torchvision.datasets.ImageFolder("archive/chest_xray/chest_xray/test",transform=transform,)
    Unkown_dataloader = DataLoader(Unkown_Dataset, shuffle=True, batch_size=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modules.AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    net.load_state_dict(torch.load('AE_2025_7_9_3/model_final.pth'))
    net.eval()

    print("Model loaded successfully")
    
    # 初始化性能指标统计
    TP, FP, TN, FN = 0, 0, 0, 0
    mse_values = []
    labels_list = []
    
    # 设定MSE阈值 - 实际应用中应在验证集上确定最佳阈值
    mse_threshold = 0.001336
    
    print(f"Starting evaluation with threshold={mse_threshold:.4f}")
    total_samples = len(Unkown_dataloader)
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(Unkown_dataloader):
            data = data.to(device)
            labels = labels.to(device)
            
            # 通过自编码器
            recon, _ = net(data)
            
            # 计算重建误差
            mse = criterion(recon, data).item()
            mse_values.append(mse)
            labels_list.append(labels.item())
            
            # 根据MSE进行分类
            predicted_class = 1 if mse >= mse_threshold else 0  # PNEUMONIA=1, NORMAL=0
            
            # 更新混淆矩阵统计
            if labels.item() == 1:  # 实际为PNEUMONIA
                if predicted_class == 1:
                    TP += 1
                else:
                    FN += 1
            else:  # 实际为NORMAL
                if predicted_class == 0:
                    TN += 1
                else:
                    FP += 1
            
            # 进度显示
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{total_samples} samples...")
    
    # 计算性能指标
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # 召回率（敏感度）
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    
    # 可视化部分：MSE分布
    plt.figure(figsize=(10, 6))
    plt.hist([mse for mse, label in zip(mse_values, labels_list) if label == 0], 
             alpha=0.7, label='NORMAL', bins=30)
    plt.hist([mse for mse, label in zip(mse_values, labels_list) if label == 1], 
             alpha=0.7, label='PNEUMONIA', bins=30)
    plt.axvline(mse_threshold, color='r', linestyle='--', label=f'Threshold ({mse_threshold})')
    plt.xlabel('MSE Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction MSE')
    plt.legend()
    plt.savefig('mse_distribution.png')
    plt.close()
    
    print("\nVisualization saved as 'mse_distribution.png'")

if __name__ == "__main__":
    main()  
    
    