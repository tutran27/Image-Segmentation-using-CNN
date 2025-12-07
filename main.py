import modal
import torch 
import torch.nn as nn
import torch.optim as optim
import os

# Các import local
from dataset import load_train_set
from img_show import show_image
from data_loader import Loader
from model import Segmentation_with_CNN
from Trainer import Trainer

#1. Tạo volume tên ..... để lưu data, sau này load lại đỡ tốn tg
volume = modal.Volume.from_name("my-segmentation-data", create_if_missing=True)

DATA_DIR = "/data"

# 2. Tạo một Modal app
app = modal.App("segmentation-project")


image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
    )
    
    .add_local_dir(
        ".",              # thư mục hiện tại (project)
        remote_path="/root",  # trong container
    )
)

@app.function(
    image=image, 
    gpu='L4', 
    volumes={DATA_DIR: volume}, 
    timeout=3600 
)
def train():
    print(f"Training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading dataset...")
    # Lưu ý: Đảm bảo load_train_set tải/đọc dữ liệu từ DATA_DIR
    train_set, test_set = load_train_set() 
    
    train_loader, test_loader = Loader(train_set, test_set)

    model = Segmentation_with_CNN(n_channels=3, n_classes=3)
    model = model.to(device)
    
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_max = 20

    print("Start Training...")
    trained_model = Trainer(model, criterion, optimizer, train_set, train_loader, test_loader, epoch_max, lr, device)
    
 

@app.local_entrypoint()
def main():
    print("Deploying to Modal...")
    train.remote()
    print("Training finished remotely.")