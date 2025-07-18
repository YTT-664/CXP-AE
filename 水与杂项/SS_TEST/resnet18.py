import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 修正数据集路径 - 使用当前环境中的正确路径
# DATA_PATH = "/home/CXP_AE/archive/chest_xray/chest_xray"  # 原始错误路径
DATA_PATH = "/root/autodl-tmp/CXP-AE/archive/chest_xray/chest_xray"  # 修改为正确的路径

# 自定义数据集类
class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        """
        data_dir: 数据集根目录
        mode: 'train', 'val', 或 'test'
        transform: 数据增强变换
        """
        self.data_dir = os.path.join(data_dir, mode)
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.samples = []
        
        # 添加路径检查
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")
        
        # 遍历目录收集样本
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            
            # 添加类目录检查
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"类别目录不存在: {class_dir}")
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # 添加样本数量检查
        if len(self.samples) == 0:
            raise RuntimeError(f"在 {self.data_dir} 中没有找到任何图像样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    # 创建数据集和数据加载器
    train_dataset = ChestXRayDataset(DATA_PATH, mode='train', transform=train_transform)
    val_dataset = ChestXRayDataset(DATA_PATH, mode='val', transform=val_transform)
    test_dataset = ChestXRayDataset(DATA_PATH, mode='test', transform=val_transform)

    # 打印数据集信息
    print(f"数据集根目录: {DATA_PATH}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型 (使用预训练ResNet-18)
    model = resnet18(pretrained=False)  # 关闭预训练

    # 修改最后一层全连接层 (二分类任务)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # 使用GPU如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 数据加载器 (batch_size=200)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)

    # 训练函数
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # 验证函数
    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, all_labels, all_preds

    # 训练参数
    num_epochs = 200  # 增加到200个epoch
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 训练循环
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  Epoch {epoch+1}: 保存最佳模型，验证准确率: {val_acc:.4f}")

    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_labels, test_preds = validate(model, test_loader, criterion, device)
    print(f"测试集准确率: {test_acc:.4f}")

    # 生成分类报告和混淆矩阵
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=['NORMAL', 'PNEUMONIA']))

    print("\n混淆矩阵:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # 可视化一些预测结果
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.axis('off')

    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    for i in range(min(8, images.size(0))):
        plt.subplot(2, 4, i+1)
        imshow(images[i].cpu())
        plt.title(f"实际: {'正常' if labels[i].item() == 0 else '肺炎'}\n预测: {'正常' if preds[i].item() == 0 else '肺炎'}")
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

except Exception as e:
    print(f"发生错误: {str(e)}")
    print("当前工作目录:", os.getcwd())
    print("尝试列出数据集目录:")
    try:
        print(os.listdir(DATA_PATH))
        for mode in ['train', 'val', 'test']:
            mode_path = os.path.join(DATA_PATH, mode)
            print(f"{mode} 目录内容:", os.listdir(mode_path))
    except Exception as e2:
        print(f"无法列出目录内容: {str(e2)}")