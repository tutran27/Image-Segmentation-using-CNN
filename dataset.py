from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F 

def target_transform(mask):
    mask=transforms.Resize((128,128))(mask)
    mask=torch.tensor(np.array(mask))
    mask-=1
    mask=mask.to(torch.long)
    return mask

def load_train_set():
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    train_set=OxfordIIITPet(root='./data',split='trainval', target_transform = target_transform, target_types='segmentation',transform=transform,download=True)
    test_set=OxfordIIITPet(root='./data',split='test',target_types='segmentation', target_transform = target_transform, transform=transform,download=True)
    return train_set,test_set