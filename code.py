# ==============================
# PREPROCESSING CODE
# *******************************


import os
import cv2
import numpy as np

INPUT_FOLDER = "raw_original_35"
OUTPUT_FOLDER = "processed_64x64"
IMAGE_SIZE = 64

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    # Convert BGR to HSV to detect blue pen ink
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue ink color range
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean small noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Slightly thicken alphabet strokes and dots
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of blue ink only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"No blue alphabet found in: {image_path}")
        return None

    # Combine all contours because Urdu letters may have separate dots
    x_min, y_min = 99999, 99999
    x_max, y_max = 0, 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore tiny noise
        if w * h < 5:
            continue

        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if x_max <= x_min or y_max <= y_min:
        print(f"Could not crop properly: {image_path}")
        return None

    # Padding around alphabet
    padding = 25
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, mask.shape[1])
    y_max = min(y_max + padding, mask.shape[0])

    cropped = mask[y_min:y_max, x_min:x_max]

    # Resize while preserving aspect ratio
    h, w = cropped.shape
    scale = (IMAGE_SIZE - 12) / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Final black background with white alphabet
    final_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    x_offset = (IMAGE_SIZE - new_w) // 2
    y_offset = (IMAGE_SIZE - new_h) // 2

    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return final_img

def main():
    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_files.sort()

    print(f"Total images found: {len(image_files)}")

    for file_name in image_files:
        input_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, file_name)

        processed = preprocess_image(input_path)

        if processed is not None:
            cv2.imwrite(output_path, processed)
            print(f"Processed: {file_name}")

    print("Preprocessing completed.")

if __name__ == "__main__":
    main()









    ==============================
TRAINING AND EVALUATION CODE
==============================

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Basic Settings
# -----------------------------
DATA_FOLDER = "processed_64x64"
RESULTS_FOLDER = "results"
IMAGE_SIZE = 64

# Updated settings to reduce memorization effect
TRAIN_AUG_PER_CLASS = 55
VAL_AUG_PER_CLASS = 12
TEST_AUG_PER_CLASS = 20

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "alif", "bay", "pay", "tay", "ttay", "say", "jeem", "chay", "hay", "khay",
    "daal", "ddaal", "zaal", "ray", "array", "zay", "zhay", "seen", "sheen",
    "suad", "zuad", "toay", "zoay", "ain", "ghain", "fay", "qaaf", "kaaf",
    "gaaf", "laam", "meem", "noon", "wow", "hay2", "yeh"
]


# -----------------------------
# Augmentation Function
# -----------------------------
def augment_image(img, mode="train"):
    h, w = img.shape

    if mode == "train":
        # Training data: moderate variations
        angle = np.random.uniform(-7, 7)
        scale = np.random.uniform(0.92, 1.08)
        tx = np.random.uniform(-3, 3)
        ty = np.random.uniform(-3, 3)
        noise_std = 3

    elif mode == "val":
        # Validation data: slightly stronger variations
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.88, 1.12)
        tx = np.random.uniform(-4, 4)
        ty = np.random.uniform(-4, 4)
        noise_std = 5

    else:
        # Test data: harder but still realistic mobile-camera style variations
        angle = np.random.uniform(-17, 17)
        scale = np.random.uniform(0.78, 1.22)
        tx = np.random.uniform(-7, 7)
        ty = np.random.uniform(-7, 7)
        noise_std = 9

    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    augmented = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Blur is mostly applied to test data to simulate mobile camera softness
    if mode == "test" and np.random.rand() > 0.55:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    # Random noise
    if np.random.rand() > 0.60:
        noise = np.random.normal(0, noise_std, augmented.shape).astype(np.float32)
        augmented = augmented.astype(np.float32) + noise
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)

    # Stroke thickness variation
    kernel = np.ones((2, 2), np.uint8)
    r = np.random.rand()

    if mode == "test":
        if r < 0.22:
            augmented = cv2.dilate(augmented, kernel, iterations=1)
        elif r < 0.44:
            augmented = cv2.erode(augmented, kernel, iterations=1)
    else:
        if r < 0.10:
            augmented = cv2.dilate(augmented, kernel, iterations=1)
        elif r < 0.20:
            augmented = cv2.erode(augmented, kernel, iterations=1)

    return augmented


# -----------------------------
# Load Original Images
# -----------------------------
def load_original_images():
    image_files = sorted([
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print("Total original processed images found:", len(image_files))

    if len(image_files) != 35:
        print("Warning: Expected 35 images, but found:", len(image_files))

    original_images = []

    for idx, file_name in enumerate(image_files):
        image_path = os.path.join(DATA_FOLDER, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Could not read:", file_name)
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Convert to binary image
        _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

        original_images.append((img, idx, file_name))

    return original_images


# -----------------------------
# Create Train, Validation, Test Sets
# -----------------------------
def create_datasets():
    original_images = load_original_images()

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for img, label, file_name in original_images:
        # Original clean image only in training
        X_train.append(img)
        y_train.append(label)

        for _ in range(TRAIN_AUG_PER_CLASS):
            X_train.append(augment_image(img, mode="train"))
            y_train.append(label)

        for _ in range(VAL_AUG_PER_CLASS):
            X_val.append(augment_image(img, mode="val"))
            y_val.append(label)

        for _ in range(TEST_AUG_PER_CLASS):
            X_test.append(augment_image(img, mode="test"))
            y_test.append(label)

    X_train = np.array(X_train, dtype="float32") / 255.0
    X_val = np.array(X_val, dtype="float32") / 255.0
    X_test = np.array(X_test, dtype="float32") / 255.0

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))
    print("Total classes:", len(np.unique(y_train)))

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# Build DNN Model
# -----------------------------
def build_dnn_model(num_classes):
    model = Sequential()

    model.add(Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))

    # DNN baseline intentionally simple hai.
    # Ye spatial features learn nahi karta, sirf flattened pixels use karta hai.
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.55))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.45))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Build CNN Model
# -----------------------------
def build_cnn_model(num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.30))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.20))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{model_name} Results")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    with open(os.path.join(RESULTS_FOLDER, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=90, values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return accuracy, precision, recall, f1


# -----------------------------
# Save Training Graphs
# -----------------------------
def save_training_graphs(history, model_name):
    name = model_name.lower()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{name}_accuracy_graph.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{name}_loss_graph.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = create_datasets()

    num_classes = len(class_names)

    X_train_model = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    X_val_model = X_val.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    X_test_model = X_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    # -----------------------------
    # DNN Model
    # -----------------------------
    print("\nTraining DNN model...")

    dnn_model = build_dnn_model(num_classes=num_classes)

    dnn_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    dnn_reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

    dnn_history = dnn_model.fit(
        X_train_model,
        y_train_cat,
        validation_data=(X_val_model, y_val_cat),
        epochs=18,
        batch_size=16,
        callbacks=[dnn_early_stop, dnn_reduce_lr],
        verbose=1
    )

    dnn_probs = dnn_model.predict(X_test_model)
    dnn_predictions = np.argmax(dnn_probs, axis=1)

    dnn_metrics = evaluate_model("DNN", y_test, dnn_predictions)
    save_training_graphs(dnn_history, "DNN")

    dnn_model.save(os.path.join(RESULTS_FOLDER, "urdu_alphabet_dnn_model.keras"))

    # -----------------------------
    # CNN Model
    # -----------------------------
    print("\nTraining CNN model...")

    cnn_model = build_cnn_model(num_classes=num_classes)

    cnn_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    cnn_reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001
    )

    cnn_history = cnn_model.fit(
        X_train_model,
        y_train_cat,
        validation_data=(X_val_model, y_val_cat),
        epochs=60,
        batch_size=16,
        callbacks=[cnn_early_stop, cnn_reduce_lr],
        verbose=1
    )

    cnn_probs = cnn_model.predict(X_test_model)
    cnn_predictions = np.argmax(cnn_probs, axis=1)

    cnn_metrics = evaluate_model("CNN", y_test, cnn_predictions)
    save_training_graphs(cnn_history, "CNN")

    cnn_model.save(os.path.join(RESULTS_FOLDER, "urdu_alphabet_cnn_model.keras"))

    # -----------------------------
    # Model Comparison Table
    # -----------------------------
    comparison = pd.DataFrame({
        "Model": ["DNN", "CNN"],
        "Accuracy": [dnn_metrics[0], cnn_metrics[0]],
        "Precision": [dnn_metrics[1], cnn_metrics[1]],
        "Recall": [dnn_metrics[2], cnn_metrics[2]],
        "F1 Score": [dnn_metrics[3], cnn_metrics[3]]
    })

    comparison.to_csv(os.path.join(RESULTS_FOLDER, "model_comparison.csv"), index=False)

    print("\nModel Comparison")
    print(comparison)

    print("\nTraining and evaluation completed.")
    print("All result files saved in results folder.")


if __name__ == "__main__":
    main()