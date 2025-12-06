🐶🐱 Image Segmentation with CNN (Oxford-IIIT Pet, PyTorch)

🧩 Task: Semantic segmentation các ảnh thú cưng (mèo / chó), dự đoán mask pixel-wise cho từng con vật + background.

📚 Dataset: Oxford-IIIT Pet

~7k ảnh mèo & chó, nhiều giống khác nhau

Sử dụng annotation segmentation (trimap) → convert về 3 lớp: background / border / pet.

🧠 Model:

Kiến trúc kiểu U-Net / encoder-decoder CNN viết bằng PyTorch

Backbone CNN trích đặc trưng, decoder upsample + skip connection.

⚙️ Training:

Data augmentation bằng torchvision.transforms

Loss: CrossEntropyLoss cho 3 lớp

Optimizer: Adam, có scheduler LR (optional)

Train / val loader dùng DataLoader với custom Dataset cho Oxford-IIIT Pet.

📈 Monitoring & Eval:

Log loss / IoU / pixel accuracy theo epoch

Lưu checkpoint tốt nhất vào models/

Notebook / script visualize: input – ground truth – predicted mask.
