from evaluate import evaluate
from predict_show import predict_show

def Trainer(model, critertion, optimizer, train_set, train_loader, test_loader, epoch_max, lr, device='cuda' ):
    img_test, mask_test=train_set[80]

    for epoch in range(epoch_max):
        total_loss=0

        for img, mask in train_loader:
            img, mask= img.to(device), mask.to(device)
            predict=model(img)
            optimizer.zero_grad()
            loss=critertion(predict, mask)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        train_loss=total_loss/len(train_loader)
        # print(f"Epoch: [{epoch+1}/{epoch_max}] --- Train Loss = {train_loss}")
        
        predict_show(img_test, mask_test, model, device)
        # Tam thoi tat de tiet kiem tinh toan
        test_loss=evaluate(model, critertion, test_loader, device)     
        print(f"Epoch: [{epoch+1}/{epoch_max}] --- Train Loss = {train_loss} ----- Test Loss = {test_loss}")
       