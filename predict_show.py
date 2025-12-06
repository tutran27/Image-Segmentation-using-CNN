import matplotlib.pyplot as plt 
import numpy as np
import torch 

def de_normalize(img, 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)):
    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)
    
    return result

def predict_show(image, mask, model, device):
  model.eval()
  img=image.to(device)
  img=img.unsqueeze(0)
  predict=model(img)
  predict=torch.argmax(predict, axis=1)
 
  image=image.numpy().transpose((1,2,0))

  predict=predict.cpu().numpy()
  predict=predict.squeeze()

  mask=mask.squeeze()

  plt.subplot(1,3,1)
  plt.title("Image")
  plt.imshow(de_normalize(image))
 

  plt.subplot(1,3,2)
  mask=mask.cpu().numpy()
  plt.title("Mask")
  plt.imshow(mask)
 

  plt.subplot(1,3,3)
  plt.imshow(predict)
  plt.title("Predict")
  plt.show()