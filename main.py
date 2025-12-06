import modal
import torch 
import torch.nn as nn
import torch.optim as optim
import os

# Các import local (đảm bảo các file này nằm cùng thư mục)
from dataset import load_train_set
from img_show import show_image
from data_loader import Loader
from model import Segmentation_with_CNN
from Trainer import Trainer

# 1. Định nghĩa Volume để lưu dữ liệu lâu dài (tránh tải lại mỗi lần chạy)
volume = modal.Volume.from_name("my-segmentation-data", create_if_missing=True)

# Đường dẫn mount volume trên cloud
DATA_DIR = "/data"
MODEL_DIR = "/data/models"

app = modal.App("segmentation-project")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
    )
)

# Gắn volume vào function
@app.function(
    image=image, 
    gpu='L4', 
    volumes={DATA_DIR: volume}, # Mount volume vào đường dẫn /data
    timeout=3600 # Tăng timeout nếu train lâu (đơn vị giây)
)
def train():
    print(f"Training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Đảm bảo thư mục lưu model tồn tại
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Sửa logic load data ---
    # Bạn cần sửa hàm load_train_set để nó nhận tham số root_dir
    # Ví dụ: train_set, test_set = load_train_set(root=DATA_DIR)
    # Nếu không sửa được file dataset.py, hãy đảm bảo nó download vào DATA_DIR
    print("Loading dataset...")
    train_set, test_set = load_train_set() 
    
    # --- Xử lý hiển thị ảnh ---
    # Không dùng plt.show() trên cloud. 
    # Nếu muốn xem, hãy lưu vào volume để tải về sau.
    # img_test, mask_test = train_set[5]
    # save_image_to_disk(img_test, f"{DATA_DIR}/sample_test.png") 

    train_loader, test_loader = Loader(train_set, test_set)

    model = Segmentation_with_CNN(n_channels=3, n_classes=3)
    model = model.to(device)
    
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_max = 20

    print("Start Training...")
    # Giả sử Trainer trả về model hoặc tự lưu
    trained_model = Trainer(model, criterion, optimizer, train_set, train_loader, test_loader, epoch_max, lr, device)
    
    # Lưu model vào Volume
    save_path = f"{MODEL_DIR}/unet_model.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Commit volume để đảm bảo dữ liệu được ghi lại an toàn
    volume.commit()

@app.local_entrypoint()
def main():
    print("Deploying to Modal...")
    train.remote()
    print("Training finished remotely.")
    
    # (Optional) Đọc file từ Volume về máy local sau khi train xong
    # Có thể dùng lệnh CLI: modal volume get my-segmentation-data /data/models/unet_model.pth .