from torch.utils.data import DataLoader

def Loader(train_set, test_set):
    train_loader=DataLoader(train_set, batch_size=32, shuffle= True, num_workers=4)
    test_loader=DataLoader(test_set, batch_size=32, shuffle= False, num_workers=4)
    return train_loader, test_loader