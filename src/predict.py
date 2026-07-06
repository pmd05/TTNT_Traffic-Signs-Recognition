import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras.models import load_model # type: ignore

from classes import CLASSES


MODEL_PATH = "../models/traffic_classifier.keras"

IMAGE_PATH = input("Enter image path: ")


model = load_model(MODEL_PATH)

image = Image.open(IMAGE_PATH)

image = image.resize((30,30))

image_array = np.array(image)

image_array = image_array.astype("float32") / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)

prediction = model.predict(image_array)

predicted_class = np.argmax(
    prediction,
    axis=1
)[0]

confidence = np.max(prediction)

print("Prediction :", CLASSES[predicted_class])

print(f"Confidence : {confidence*100:.2f}%")

original = Image.open(IMAGE_PATH)

plt.imshow(original)

plt.axis("off")

plt.title(
    f"{CLASSES[predicted_class]}\nConfidence : {confidence*100:.2f}%"
)

plt.show()