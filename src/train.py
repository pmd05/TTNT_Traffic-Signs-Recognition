import os
os.makedirs("../images", exist_ok=True)
import matplotlib.pyplot as plt
from dataset import DatasetLoader
from model import TrafficSignModel


dataset_path = "../dataset"

loader = DatasetLoader(dataset_path)

X_train, X_test, y_train, y_test = loader.load_data()

print("Train:", X_train.shape)
print("Test:", X_test.shape)

cnn = TrafficSignModel()

model = cnn.build_model()

model.summary()

EPOCHS = 15
BATCH_SIZE = 32

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

os.makedirs("../models", exist_ok=True)

model.save("../models/traffic_classifier.keras")

print("Model saved successfully!")

# Accuracy
plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Model Accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "../images/accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Loss
plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Model Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "../images/loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTraining completed successfully!")
print("Model saved to: ../models/traffic_classifier.keras")
print("Accuracy graph: ../images/accuracy.png")
print("Loss graph: ../images/loss.png")