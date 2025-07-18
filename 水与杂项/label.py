import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import ImageFile
import AE_2025_7_9_3.modules as modules 

def evaluate_pseudo_labels(model, test_loader, threshold, true_labels, device='cpu'):
    model.eval()
    model.to(device)
    reconstruction_errors = []
    predicted_labels = []

    with torch.no_grad():
        for data in test_loader:
            x, _ = data  
            x = x.to(device).to(torch.float32)

            x_hat = model(x)
            if isinstance(x_hat, tuple):  
                x_hat = x_hat[0]

            loss = F.mse_loss(x_hat, x, reduction='none')  
            loss_per_sample = loss.view(loss.size(0), -1).mean(dim=1)  
            reconstruction_errors.extend(loss_per_sample.cpu().numpy())

            pred_labels = (loss_per_sample > threshold).int()
            predicted_labels.extend(pred_labels.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels[:len(predicted_labels)]) 

    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("\n[Evaluation Results]")
    print(f"  Threshold τ     : {threshold}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall (Sens.)  : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")

    return reconstruction_errors, predicted_labels


def plot_error_distribution(errors, threshold, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=100, alpha=0.7, label='Reconstruction Errors')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold τ = {threshold:.4f}')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Sample Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] MSE Distribution plot at: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("reconstruction_comparisons", exist_ok=True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose([
        modules.AddGaussianNoise(amplitude=15),
        transforms.Resize([1056, 1056]),
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.ImageFolder("/root/autodl-tmp/CXP-AE/archive/chest_xray/chest_xray/train", transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules.AutoEncoder().to(device)
    model.load_state_dict(torch.load("/root/autodl-tmp/CXP-AE/AE_2025_7_9_3/model_final.pth"))
    print("[Loaded] AutoEncoder model.")

    true_labels = [label for _, label in test_dataset.samples]

    mse_threshold = 0.001336

    errors, preds = evaluate_pseudo_labels(model, test_loader, threshold=mse_threshold, true_labels=true_labels, device=device)

    plot_error_distribution(errors, threshold=mse_threshold, save_path="loss_plots/mse_distribution.png")
    

if __name__ == "__main__":
    main()
