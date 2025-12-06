import torch

def evaluate(model, criterion, test_loader, device):
    model.eval()
    loss_total=0
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask= img.to(device), mask.to(device)
            out=model(img)
            loss=criterion(out, mask)
            loss_total+=loss.item()
        loss_test=loss_total.len(test_loader)

    return loss_test
