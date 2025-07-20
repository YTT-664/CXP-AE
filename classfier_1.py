import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.io import read_image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.samples = []
        
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpeg') or img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = read_image(img_path).float() / 255.0
        
       
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MSCAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(MSCAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        ca = self.channel_attention(x)
        
        sa = self.spatial_attention(x)
        
        att = ca * sa
        return x * att + x


class IAFF(nn.Module):
    def __init__(self, in_channels):
        super(IAFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.mscam = MSCAM(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.mscam(x)
        x = self.conv2(x)
        return self.silu(x)

class ModifiedStem(nn.Module):
    def __init__(self):
        super(ModifiedStem, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.silu = nn.SiLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.branch_conv1 = nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=False)
        self.branch_conv2 = nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=False)
        self.branch_bn = nn.BatchNorm2d(32)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.maxpool(x)
        
        branch_left = self.branch_conv1(x)
        branch_left = self.branch_conv2(branch_left)
        branch_left = self.branch_bn(branch_left)
     
        branch_right = self.branch_conv1(x)
        branch_right = self.branch_bn(branch_right)

        x = torch.cat([branch_left, branch_right], dim=1)
        return self.silu(x)

class ImprovedInceptionResNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedInceptionResNetV2, self).__init__()

        self.stem = ModifiedStem()

        self.inception_a = self._make_inception_a(64)
        self.reduction_a = self._make_reduction_a(64)
        
        self.inception_b = self._make_inception_b(128)
        self.reduction_b = self._make_reduction_b(128)
        
        self.inception_c = self._make_inception_c(256)

        self.iaff1 = IAFF(64)
        self.iaff2 = IAFF(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):

        x_stem = self.stem(x)

        x_a = self.inception_a(x_stem)

        x_fused1 = self.iaff1(x_stem, x_a)

        x_red_a = self.reduction_a(x_fused1)

        x_b = self.inception_b(x_red_a)

        x_fused2 = self.iaff2(x_red_a, x_b)

        x_red_b = self.reduction_b(x_fused2)

        x_c = self.inception_c(x_red_b)

        x = self.avgpool(x_c)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def _make_inception_a(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
    
    def _make_reduction_a(self, in_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
    
    def _make_inception_b(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
    
    def _make_reduction_b(self, in_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
    
    def _make_inception_c(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            MSCAM(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if i % 100 == 99:
            print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f'Epoch {epoch} Train - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}')
    return accuracy, loss.item()

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    val_loss /= len(val_loader)
    print(f'Epoch {epoch} Validation - Loss: {val_loss:.4f}, Acc: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    return accuracy, val_loss

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print('Test Results:')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    return all_probs, all_labels

def main():
    data_dir = 'data'  
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = PneumoniaDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = PneumoniaDataset(os.path.join(data_dir, 'val'), transform=test_transform)
    test_dataset = PneumoniaDataset(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ImprovedInceptionResNetV2(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, val_loss = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with val acc: {best_val_acc:.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    test_probs, test_labels = test(model, test_loader, device)

    np.save('test_probs.npy', test_probs)
    np.save('test_labels.npy', test_labels)

if __name__ == '__main__':
    main()