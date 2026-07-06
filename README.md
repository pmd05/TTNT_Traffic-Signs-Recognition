# 🚦 Traffic Signs Recognition using CNN & TensorFlow

## 📌 Giới thiệu

Đây là đồ án nhận diện biển báo giao thông sử dụng **Convolutional Neural Network (CNN)** được xây dựng bằng **TensorFlow/Keras**.

Chương trình có thể:

- Huấn luyện mô hình CNN từ bộ dữ liệu GTSRB
- Đánh giá mô hình trên tập kiểm tra
- Nhận diện một ảnh biển báo bất kỳ
- Giao diện đồ họa (GUI) bằng Tkinter
- Hiển thị tên biển báo và độ tin cậy (Confidence)

---

# 📂 Cấu trúc Project

```
Traffic-Signs-Recognition/
│
├── dataset/
│   ├── Train/
│   ├── Test/
│   ├── Meta/
│   └── Test.csv
│
├── models/
│   └── traffic_classifier.keras
│
├── images/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gui.py
│   ├── classes.py
│   └── utils.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── main.py
```


# 📥 Cài đặt
## 1. Tải xuống dataset và đưa vào thư mục dataset/

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Cấu trúc Folder:
dataset/
├── Train/
├── Test/
├── Meta/
├── Meta.csv
├── Train.csv
└── Test.csv

## 2. Tạo Virtual Environment

Windows

```bash
python -m venv venv
```

Kích hoạt

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# 🚀 Huấn luyện mô hình

Di chuyển vào thư mục

```bash
cd src
```

Sau đó chạy

```bash
python train.py
```

Sau khi hoàn thành sẽ sinh ra:

```
models/
└── traffic_classifier.keras
```

và

```
images/

accuracy.png

loss.png
```

---

# 📈 Đánh giá mô hình

Sau khi đã train xong

```bash
python evaluate.py
```

Kết quả:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

Đồng thời lưu:

```
images/
└── confusion_matrix.png
```

---

# 🖼 Dự đoán một ảnh

```bash
python predict.py
```

Ví dụ kết quả

```
Prediction : Stop

Confidence : 99.52%
```

---

# 🖥 Chạy giao diện GUI

Sau khi đã có model

```bash
python gui.py
```

Giao diện sẽ xuất hiện.

Các bước sử dụng:

1. Nhấn **Upload Image**
2. Chọn ảnh biển báo
3. Chương trình tự động nhận diện
4. Hiển thị:
   - Hình ảnh
   - Tên biển báo
   - Confidence (%)

---

# 📁 Kết quả sinh ra

```
models/

traffic_classifier.keras
```

```
images/

accuracy.png

loss.png

confusion_matrix.png
```

---

# 📚 Công nghệ sử dụng

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn
- Tkinter

---

# 🧠 Kiến trúc CNN

```
Input (30×30×3)

↓

Conv2D (32)

↓

Conv2D (32)

↓

MaxPooling

↓

Dropout

↓

Conv2D (64)

↓

Conv2D (64)

↓

MaxPooling

↓

Dropout

↓

Flatten

↓

Dense (256)

↓

Dropout

↓

Dense (43)

↓

Softmax
```

---

# 📊 Đầu ra

Chương trình trả về:

- Tên biển báo
- Confidence (%)

Ví dụ

```
Prediction

Stop

Confidence

99.82%
```

---

# ⚠ Lưu ý

- Không đổi cấu trúc thư mục của dataset.
- Chỉ chạy `gui.py` sau khi đã train và có file:

```
models/traffic_classifier.keras
```

- Nếu gặp lỗi thiếu thư viện:

```bash
pip install -r requirements.txt
```

---


# 📖 Tài liệu tham khảo

- TensorFlow Documentation
- Keras Documentation
- German Traffic Sign Recognition Benchmark (GTSRB)
- DataFlair - Traffic Signs Recognition using CNN