# 🐶🐱 Image Segmentation with CNN (Oxford-IIIT Pet, PyTorch)

## 🚀 Overview
Mô hình segmentation thú cưng (mèo/chó) dùng CNN, triển khai bằng **PyTorch**.

## 📚 Dataset
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- 3 lớp: background / border / pet.

## 🧠 Model
- Kiến trúc encoder–decoder kiểu U-Net.
- Loss: `CrossEntropyLoss`, optimizer: `Adam`.

## ⚙️ Usage

```bash
pip install -r requirements.txt

