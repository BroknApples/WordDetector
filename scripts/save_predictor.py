"""
Construct a CRNN OCR model from pretrained weights and save it as a .keras predictor.
Includes convolutional, BiLSTM, and dense layers for character recognition.
"""


import sys
import os
from typing import Final
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import layers # type: ignore


# -------------------- Constants --------------------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256
CHARS: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"
MAX_LABEL_LEN: Final = 16
NUM_CLASSES: Final = len(CHARS) + 1  # +1 for CTC blank

WEIGHTS_PATH: Final = "models/crnn.weights.h5"
PREDICTOR_PATH: Final = "models/crnn_predictor.keras"


def main():
  """
  Entry point for the program.

  Build the prediction model and save it as a .keras file

  Parameters
  ----------
    argc : Number of command-line arguments provided.
    argv : List of command-line arguments. (NONE).

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
  
  -------
  """

  # -------------------- Load Weights and Save Predictor --------------------
  if not os.path.exists(WEIGHTS_PATH):
    print(f"Weights file not found at {WEIGHTS_PATH}")
    return -1

  input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

  # Conv layers
  x = layers.Conv2D(64, 3, padding="same", activation="relu")(input_img)
  x = layers.MaxPooling2D((2,2))(x)
  x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
  x = layers.MaxPooling2D((2,2))(x)

  # Collapse height
  pool_h, pool_w = x.shape[1], x.shape[2]
  x = layers.Reshape(target_shape=(pool_w, pool_h * 128))(x)

  # BiLSTM
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

  # Output
  y_pred = layers.Dense(NUM_CLASSES, activation="softmax", name="y_pred")(x)

  # Load and save model
  model = keras.Model(inputs=input_img, outputs=y_pred, name="crnn_predictor")
  model.load_weights(WEIGHTS_PATH)
  model.save(PREDICTOR_PATH)

  print(f"Prediction model saved at: {PREDICTOR_PATH}")

  return 0


if __name__ == "__main__":
  print("")
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
