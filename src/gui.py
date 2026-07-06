import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
from classes import CLASSES

MODEL_PATH = "../models/traffic_classifier.keras"
model = load_model(MODEL_PATH)

root = tk.Tk()
root.title("Traffic Sign Recognition")

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - WINDOW_WIDTH) // 2
y = (screen_height - WINDOW_HEIGHT) // 2

root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

root.configure(bg="#f2f2f2")

root.resizable(False, False)


title = tk.Label(
    root,
    text="Traffic Sign Recognition",
    font=("Arial", 30, "bold"),
    bg="#f2f2f2"
)

title.pack(pady=20)

main_frame = tk.Frame(
    root,
    bg="#f2f2f2"
)

main_frame.pack(
    expand=True,
    fill="both",
    padx=40,
    pady=20
)

left_frame = tk.Frame(
    main_frame,
    bg="white",
    relief="ridge",
    bd=2,
    width=500,
    height=500
)

left_frame.pack(
    side="left",
    padx=20,
    pady=20
)

left_frame.pack_propagate(False)


image_label = tk.Label(
    left_frame,
    text="Image Preview",
    font=("Arial",18),
    bg="white"
)

image_label.pack(
    expand=True
)

right_frame = tk.Frame(
    main_frame,
    bg="white",
    relief="ridge",
    bd=2,
    width=450,
    height=500
)

right_frame.pack(
    side="right",
    padx=20,
    pady=20,
    fill="y"
)

right_frame.pack_propagate(False)

prediction_title = tk.Label(

    right_frame,

    text="Prediction",

    font=("Arial",20,"bold"),

    bg="white"

)

prediction_title.pack(
    pady=(50,10)
)

result_label = tk.Label(

    right_frame,

    text="Waiting for image...",

    font=("Arial",18),

    fg="blue",

    bg="white",

    wraplength=350,

    justify="center"

)

result_label.pack()

confidence_title = tk.Label(

    right_frame,

    text="Confidence",

    font=("Arial",20,"bold"),

    bg="white"

)

confidence_title.pack(
    pady=(40,10)
)


confidence_label = tk.Label(

    right_frame,

    text="0.00%",

    font=("Arial",18),

    fg="green",

    bg="white"

)

confidence_label.pack()

separator = tk.Frame(

    right_frame,

    height=2,

    bg="#d9d9d9"

)

separator.pack(

    fill="x",

    padx=30,

    pady=40

)

def predict_image(file_path):

    # Đọc ảnh gốc
    image = Image.open(file_path).convert("RGB")

    # Chuẩn bị ảnh cho CNN
    resized = image.resize((30, 30))

    img = np.array(resized)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)

    # Hiển thị kết quả
    result_label.config(
        text=CLASSES[class_id]
    )

    confidence_label.config(
        text=f"{confidence * 100:.2f}%"
    )

    display = image.copy()

    display.thumbnail((430, 430))

    photo = ImageTk.PhotoImage(display)

    image_label.config(
        image=photo,
        text=""
    )

    image_label.image = photo


def upload_image():

    file_path = filedialog.askopenfilename(

        title="Choose Traffic Sign Image",

        filetypes=[

            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),

            ("PNG", "*.png"),

            ("JPEG", "*.jpg *.jpeg")

        ]

    )

    if file_path:

        predict_image(file_path)


upload_button = tk.Button(

    right_frame,

    text="Upload Image",

    command=upload_image,

    font=("Arial", 18, "bold"),

    bg="#0078D7",

    fg="white",

    activebackground="#005A9E",

    activeforeground="white",

    cursor="hand2",

    width=18,

    height=2

)

upload_button.pack(pady=20)

exit_button = tk.Button(

    right_frame,

    text="Exit",

    command=root.destroy,

    font=("Arial", 14),

    width=18,

    bg="#E74C3C",

    fg="white",

    cursor="hand2"

)

exit_button.pack()

footer = tk.Label(

    root,

    text="Traffic Sign Recognition using CNN & TensorFlow",

    font=("Arial", 10),

    bg="#f2f2f2",

    fg="gray"

)

footer.pack(pady=10)

root.mainloop()