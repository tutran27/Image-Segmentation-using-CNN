# import modal
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from dataset import load_train_set
from img_show import show_image
from data_loader import Loader
from model import Segmentation_with_CNN
from evaluate import evaluate
from Trainer import Trainer

# app=modal.App()

# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch",
#         "torchvision",
#         "numpy",
#         "matplotlib",
#     )
# )

# @app.function(image=image, gpu='L4')
# def train():
    # Load dataset
device='cuda' if torch.cuda.is_available() else 'cpu'

train_set, test_set = load_train_set()
img_test, mask_test = train_set[5]
show_image(img_test, mask_test)

train_loader, test_loader=Loader(train_set, test_set)

model=Segmentation_with_CNN(n_channels=3, n_classes=3)

    # Set model
model=model.to(device)
lr=1e-3
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=lr)

epoch_max=20
    

Trainer(model, criterion, optimizer,train_set, train_loader, test_loader,epoch_max, lr, device)

# @app.local_entrypoint()
# def main():
#     # gọi remote -> chạy trên Modal, bắt đầu trừ credits
#     train.remote()




