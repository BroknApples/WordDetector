"""
EAST Text Detection

This module provides functions to detect text regions in images using
the EAST (Efficient and Accurate Scene Text) deep learning model.

Functions:
----------
- _decode(scores, geometry, thresh): Decode EAST model outputs into
    bounding boxes and confidence scores. Private function.
- detectTextBBFromImage(path, size_type, show_img, box_color, box_thickness):
    Detect text in a single image and return bounding boxes.
- detectTextBBFromImages(paths, size_type, show_img, box_color, box_thickness):
    Detect text in multiple images, returning a list of bounding boxes for each image.

Usage:
------
This module is not meant to be run directly. Import and use its
functions in other scripts to perform text detection.
"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT = 32
IMG_WIDTH = 256

def buildTextClassifier(num_classes: int):
  model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes, activation='softmax')
  ])

  model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  return model



# ---------- Load Dataset ----------
train_dir = "data/my_dataset/train"
val_dir   = "data/my_dataset/val"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    label_mode="int"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    label_mode="int"
)

# For speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# Number of words = number of folders
num_classes = len(train_ds.class_names)
print("Detected classes:", train_ds.class_names)
print("Num classes:", num_classes)

# ---------- Train ----------
model = buildTextClassifier(num_classes)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)

# ---------- Save model ----------
model.save("word_classifier.h5")

print("Training complete. Model saved as word_classifier.h5")