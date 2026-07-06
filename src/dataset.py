import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.num_classes = 43
        self.image_size = (30, 30)

    def load_data(self):
        data = []
        labels = []
        train_path = os.path.join(self.dataset_path, "Train")

        # Lặp qua từng class
        for class_id in range(self.num_classes):
            class_path = os.path.join(train_path, str(class_id))

            # Lặp qua từng ảnh trong class
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path)
                image = image.resize(self.image_size)
                image = np.array(image)

                data.append(image)
                labels.append(class_id)

        # Chuyển sang numpy array
        data = np.array(data)
        labels = np.array(labels)

        # Chuẩn hóa dữ liệu
        data = data.astype("float32") / 255.0

        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # One-hot encoding cho nhãn
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        return X_train, X_test, y_train, y_test
