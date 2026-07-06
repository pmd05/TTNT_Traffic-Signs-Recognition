import os
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model # type: ignore


MODEL_PATH = "../models/traffic_classifier.keras"
TEST_CSV = "../dataset/Test.csv"

model = load_model(MODEL_PATH)

print("Model loaded!")

test_df = pd.read_csv(TEST_CSV)

labels = test_df["ClassId"].values
paths = test_df["Path"].values

images = []

for path in paths:

    image_path = os.path.join("../dataset", path)

    image = Image.open(image_path)

    image = image.resize((30,30))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    images.append(image)

X_test = np.array(images)

predictions = model.predict(X_test)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

accuracy = accuracy_score(
    labels,
    predicted_labels
)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")

print(
    classification_report(
        labels,
        predicted_labels
    )
)

cm = confusion_matrix(
    labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap="Blues", xticks_rotation=90)

plt.tight_layout()

plt.savefig(
    "../images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()