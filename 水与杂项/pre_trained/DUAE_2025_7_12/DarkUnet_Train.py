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

def main():
    
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("reconstruction_comparisons", exist_ok=True)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        transforms.Resize([416, 416]),
        transforms.ToTensor(),
    ])
    Train_Dataset = modules.DUAE_Dataset(
        root_dir="archive/chest_xray/chest_xray/train/NORMAL",
        transform=transform,
        should_invert=False,
        n=700,
        noice_amplitude=51
    )
    train_dataloader = DataLoader(
        Train_Dataset, 
        shuffle=True, 
        batch_size=48
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modules.DarkUnet().to(device)
    #criterion = nn.SmoothL1Loss(beta=0.04)
    #criterion =nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    
    # 初始化记录loss的列表[2,3](@ref)
    epoch_losses = []  # 记录每个epoch的平均loss
    batch_losses = []  # 记录每个batch的loss（用于更详细的曲线）
    all_epochs = []    # 记录epoch数
    fixed_pairs = []  # 存储(噪声图, 原图)对
    sample_count = 0
    for data in train_dataloader:
        if sample_count >= 10:
            break
        # 同时保存噪声图和原图
        fixed_pairs.append((data[0][:1], data[1][:1]))
        sample_count += 1

    # 分离为两个张量
    fixed_noisy = torch.cat([p[0] for p in fixed_pairs], dim=0).to(device)
    fixed_clean = torch.cat([p[1] for p in fixed_pairs], dim=0).to(device)
    
    for epoch in range(600):
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
            # 记录每个batch的loss[2](@ref)
            batch_losses.append(loss.item())
            
            # 每10个batch打印一次进度
            if batch_id % 10 == 0:
                avg_loss = train_loss / (batch_id + 1)
                print(f"Epoch {epoch+1}/100 | Batch {batch_id} | Loss: {avg_loss:.4f}")
        
        # 计算并记录本epoch的平均loss[3](@ref)
        epoch_time = time.time() - start_time
        avg_loss = train_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        all_epochs.append(epoch + 1)
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # 每5个epoch保存一次模型和loss曲线图
        if (epoch + 1) % 5 == 0:
            # 保存模型
            model_path = f"pth/model_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # 生成并保存loss走势图[3,4](@ref)
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
            loss_plot_path = f"loss_plots/loss_plot_epoch_{epoch+1}.png"
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved to {loss_plot_path}")
            
        if (epoch + 1) % 5 == 0:
            # ... 原有保存模型和loss图的代码 ...
            
            # 2. 生成并保存重建对比图
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
                comparison_path = f"reconstruction_comparisons/epoch_{epoch+1}.png"
                grid_image.save(comparison_path)
            
            net.train()
    
    # 保存最终模型
    final_path = f"pth/model_final.pth"
    torch.save(net.state_dict(), final_path)
    print(f"Training completed. Final model saved to {final_path}")
    
    # 训练结束后保存最终loss曲线图[9](@ref)
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, epoch_losses, 'b-o')
    plt.title('Final Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("loss_plots/final_loss_plot.png")
    plt.close()
    print("Final loss plot saved.")

if __name__ == "__main__":
    main()