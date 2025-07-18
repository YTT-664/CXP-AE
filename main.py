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

MODE             = 1 
# 0:NO LABEL SAMPLE  
# 1:SMALL LABEL SAMPLE  
# 2:generate images of the estimate distribution, for helping deciding what values of OFFSET, LAMBDA1 and LAMBDA1 is.
#       blue line = normal 
#       red line  = pneumonia
#       green line= normal/pneumonia
#       yellowline= pneumonia/normal
EPOCH            = 400
SIZE             = 416
NOICE_A          = 26
OFFSET           = 52
PRIORI_P         = 0.5

TRAIN_N          = 250
BATCH_SIZE       = 50

LAMBDA1          = 2.5
LAMBDA2          = 1.25


OTHER_DATA_ROOT  = "Other_Data"
SMALL_SAMPLE_ROOT= "archive/chest_xray/train/NORMAL"
DATASET_ROOT     = "archive/chest_xray/train/UNKOWN"
PTH_ROOT         = "pth"
P_P_ROOT         = "Pseudo_Data/PNEUMONIA"
P_N_ROOT         = "Pseudo_Data/NORMAL"
P_U_ROOT         = "Pseudo_Data/UNKOWN"


def pretrain(EPOCH,Size,Noice_A,Root,Train_N,Batch_Size,pth_ROOT,O_Root):
    
    os.makedirs(O_Root+"/loss_plots", exist_ok=True)
    os.makedirs(O_Root+"/reconstruction_comparisons", exist_ok=True)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        transforms.Resize([Size, Size]),
        transforms.ToTensor(),
    ])
    Train_Dataset = modules.DUAE_Dataset(
        root_dir=Root,
        transform=transform,
        should_invert=False,
        n=Train_N,
        noice_amplitude=Noice_A
    )
    train_dataloader = DataLoader(
        Train_Dataset, 
        shuffle=True, 
        batch_size=Batch_Size,
        num_workers=32,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modules.DarkUnet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    
    # 初始化记录loss的列表
    epoch_losses = []  # 记录每个epoch的平均loss
    batch_losses = []  # 记录每个batch的loss（用于更详细的曲线）
    all_epochs = []    # 记录epoch数
    fixed_pairs = []  # 存储(噪声图, 原图)对
    sample_count = 0
    for data in train_dataloader:
        if sample_count >= 10:
            break
        fixed_pairs.append((data[0][:1], data[1][:1]))
        sample_count += 1

    # 分离为两个张量
    fixed_noisy = torch.cat([p[0] for p in fixed_pairs], dim=0).to(device)
    fixed_clean = torch.cat([p[1] for p in fixed_pairs], dim=0).to(device)
    
    for epoch in range(EPOCH):
        net.train()
        train_loss = 0.0
        start_time = time.time()
        

        
        for batch_id, (dataN,dataO) in enumerate(train_dataloader):
            dataN = dataN.to(device)
            dataO = dataO.to(device)
            optimizer.zero_grad()
            output= net(dataN)
            loss = criterion(dataO, output)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # 记录每个batch的loss
            batch_losses.append(loss.item())
            
            # 每10个batch打印一次进度
            if batch_id % 10 == 0:
                avg_loss = train_loss / (batch_id + 1)
                print(f"Epoch {epoch+1}/{EPOCH} | Batch {batch_id} | Loss: {avg_loss:.4f}")
        
        # 计算并记录本epoch的平均loss
        epoch_time = time.time() - start_time
        avg_loss = train_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        all_epochs.append(epoch + 1)
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # 每5个epoch保存一次模型和loss曲线图
        if (epoch + 1) % 5 == 0:
            # 保存模型
            model_path = pth_ROOT+f"/model_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # 生成并保存loss走势图
            plt.figure(figsize=(10, 6))
            
            # 绘制每个epoch的平均loss曲线
            plt.subplot(2, 1, 1)
            plt.plot(all_epochs, epoch_losses, 'b-', label='Epoch Loss')
            plt.title(f'Training Loss Curve (Epoch {epoch+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 绘制更详细的batch loss曲线（最近500个batch）
            plt.subplot(2, 1, 2)
            recent_batches = min(500, len(batch_losses))
            batch_range = list(range(len(batch_losses) - recent_batches + 1, len(batch_losses) + 1))
            plt.plot(batch_range, batch_losses[-recent_batches:], 'r-', alpha=0.7, label='Batch Loss')
            plt.title('Recent Batch Losses')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存loss图片
            loss_plot_path = O_Root+f"/loss_plots/loss_plot_epoch_{epoch+1}.png"
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved to {loss_plot_path}")
            
        if (epoch + 1) % 5 == 0:
            # 原有保存模型和loss图的代码
            
            # 生成并保存重建对比图
            net.eval()
            with torch.no_grad():
                reconstructions = net(fixed_noisy)
                
                # 三合一对比
                comparison = torch.cat([
                    fixed_clean, 
                    fixed_noisy, 
                    reconstructions
                ], dim=0)
                
                grid = torchvision.utils.make_grid(
                    comparison.cpu().data,
                    nrow=sample_count, 
                    normalize=True,
                    scale_each=True
                )
                
                # 保存为图像文件
                grid_image = transforms.ToPILImage()(grid)
                comparison_path = O_Root+f"/reconstruction_comparisons/epoch_{epoch+1}.png"
                grid_image.save(comparison_path)
            
            net.train()
    
    # 保存最终模型
    final_path = pth_ROOT+f"/model_final.pth"
    torch.save(net.state_dict(), final_path)
    print(f"Training completed. Final model saved to {final_path}")
    
    # 训练结束后保存最终loss曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, epoch_losses, 'b-o')
    plt.title('Final Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(O_Root+"/loss_plots/final_loss_plot.png")
    plt.close()
    print("Final loss plot saved.")
    
def Get_Threshold_0(Root,Noice_A,Net,Offset,P,L1,L2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=Offset),
        transforms.Resize([416,416]),
        transforms.ToTensor(),
    ])
    Dataset = modules.DUAE_Dataset(
        root_dir=Root,
        transform=transform,
        should_invert=False,
        n=2000,
        noice_amplitude=Noice_A
    )
    dataloader = DataLoader(
        Dataset, 
        shuffle=True, 
        batch_size=1,
        pin_memory=True
    )
    MAEs=modules.Only_evaluate_dataset(dataloader,Net,criterion,device)
    p, mu1, sigma1, mu2, sigma2=modules.mixture_mle(MAEs,p=P,max_iter=2000)
    print(p)
    modules.plot_and_save_normal_distributions(mu1,sigma1,mu2,sigma2,p,"distribution.png")
    THRESHOLD_up=mu1+L1*sigma1
    
    THRESHOLD_bottom=mu2-L2*sigma2
    
    return THRESHOLD_up,THRESHOLD_bottom

def Get_Threshold_1(Root,Noice_A,Net,Offset,P):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=Offset),
        transforms.Resize([416,416]),
        transforms.ToTensor(),
    ])
    Dataset = modules.DUAE_Dataset(
        root_dir=Root,
        transform=transform,
        should_invert=False,
        n=1000,
        noice_amplitude=Noice_A
    )
    dataloader = DataLoader(
        Dataset, 
        shuffle=True, 
        batch_size=1,
        pin_memory=True
    )
    MAEs=modules.Only_evaluate_dataset(dataloader,Net,criterion,device)
    mean = sum(MAEs) / len(MAEs)
    sigama = sum((x - mean)**2 for x in MAEs) / len(MAEs)
    THRESHOLD_up=mean+2*sigama
    THRESHOLD_bottom=mean
    return THRESHOLD_up,THRESHOLD_bottom

def Pseudo_Lable(Root,Net,Offset,Noice_A,Threshold_P,Threshold_N,P_P_Root,P_N_Root,P_U_Root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=Offset),
        transforms.Resize([416,416]),
        transforms.ToTensor(),
    ])
    Dataset = modules.DUAE_Lable_Dataset(
        root_dir=Root,
        transform=transform,
        should_invert=False,
        noice_amplitude=Noice_A,
        n=4000
    )
    dataloader = DataLoader(
        Dataset, 
        shuffle=True, 
        batch_size=1,
        pin_memory=True
    )
    modules.evaluate_dataset(dataloader,Net,criterion,device,Threshold_P,Threshold_N,True,P_P_Root,P_U_Root,P_N_Root,0.023,1)

if __name__ == "__main__":
    os.makedirs(OTHER_DATA_ROOT, exist_ok=True)
    os.makedirs(PTH_ROOT, exist_ok=True)
    os.makedirs(P_P_ROOT, exist_ok=True)
    os.makedirs(P_N_ROOT, exist_ok=True)
    os.makedirs(P_U_ROOT, exist_ok=True)
    if MODE==0:
        pretrain(EPOCH,SIZE,NOICE_A,DATASET_ROOT,TRAIN_N,BATCH_SIZE,PTH_ROOT,OTHER_DATA_ROOT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net=modules.DarkUnet().to(device)
        net.load_state_dict(torch.load(PTH_ROOT+f"/model_final.pth"))
        Threshold_P,Threshold_N=Get_Threshold_0(DATASET_ROOT,NOICE_A,net,OFFSET,PRIORI_P)
        print(Threshold_P,Threshold_N)
        Pseudo_Lable(DATASET_ROOT,net,OFFSET,NOICE_A,Threshold_P,Threshold_P,P_P_ROOT,P_N_ROOT,P_U_ROOT)
    elif MODE==1:
        pretrain(EPOCH,SIZE,NOICE_A,SMALL_SAMPLE_ROOT,TRAIN_N,BATCH_SIZE,PTH_ROOT,OTHER_DATA_ROOT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net=modules.DarkUnet().to(device)
        net.load_state_dict(torch.load(PTH_ROOT+f"/model_final.pth"))
        Threshold_P,Threshold_N=Get_Threshold_0(DATASET_ROOT,NOICE_A,net,OFFSET,PRIORI_P,LAMBDA1,LAMBDA2)
        print(Threshold_P,Threshold_N)
        Pseudo_Lable(DATASET_ROOT,net,OFFSET,NOICE_A,Threshold_P,Threshold_N,P_P_ROOT,P_N_ROOT,P_U_ROOT)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net=modules.DarkUnet().to(device)
        net.load_state_dict(torch.load(PTH_ROOT+f"/model_final.pth"))
        Threshold_P,Threshold_N=Get_Threshold_0(DATASET_ROOT,NOICE_A,net,OFFSET,PRIORI_P,LAMBDA1,LAMBDA2)
        print(Threshold_P,Threshold_N)
