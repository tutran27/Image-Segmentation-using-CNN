import matplotlib.pyplot as plt

def show_image(image, mask):
    img=image.permute(1,2,0)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1,2,1)
    plt.axis('off')

    plt.imshow(mask.cpu().numpy(),cmap='gray')
    plt.title('Mask')
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.show()