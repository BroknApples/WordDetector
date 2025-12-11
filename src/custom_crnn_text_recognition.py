"""
Helper functions for using a custom CRNN model to recognize text in images.

Includes:
- Preprocessing image crops for CRNN input
- Decoding CRNN predictions using CTC
- Recognizing text from bounding boxes with a trained CRNN model

This module is not intended to be run directly.
"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import cv2
import numpy as np
from typing import Final

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # Silence some tensorflow output
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # Silence more tensorflow output

from typing import Final
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras import backend as K # type: ignore



# ---------- Constants ----------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256
CHARS: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"
NUM_CLASSES: Final = len(CHARS) + 1  # +1 for CTC blank


# ---------- Helper functions ----------
def _preprocessBox(image, box):
  """
  Crop and preprocess an image region for CRNN prediction.

  Parameters
  ----------
    image : Original image in BGR format.
    box : Bounding box coordinates (x1, y1, x2, y2).

  Returns
  -------
    np.ndarray : Preprocessed image crop, resized, normalized, and with batch & channel dimensions.

  -------
  """

  x1, y1, x2, y2 = box
  crop = image[y1:y2, x1:x2]
  crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  crop = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))  # fixed width
  crop = crop.astype("float32") / 255.0
  crop = np.expand_dims(crop, axis=(0, -1))
  return crop

def _decodePrediction(pred, chars=CHARS):
  """
  Decode a batch of CRNN predictions using CTC decoding.

  Parameters
  ----------
    pred : Predicted probabilities from the CRNN model, shape [batch, time_steps, num_classes].
    chars : String of valid characters, default is CHARS.

  Returns
  -------
    list of str : Decoded text for each sequence in the batch.
  
  -------
  """

  # pred: [batch, time_steps, num_classes]
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
  decoded = decoded[0].numpy()

  result = []
  for seq in decoded:
    text = ""
    for c in seq:
      if c == -1: continue # Blank
      elif c >= len(chars): continue
      text += chars[c]
    result.append(text)
  return result


def recognizeTextFromBBs_CustomCRNN(image: np.ndarray, boxes: list[tuple[int, int, int, int]], model):
  """
  Recognizes text given an image and a list of bounding boxes along with the trained custom crnn model
  
  Parameters
  ----------
    image : Image to read
    boxes : List of bounding-box coordinates on the image
    model : Trained CRNN model from 'train_crnn.py'

  Returns
  -------
    result: List of results
  
  -------
  """

  results = []
  for box in boxes:
    crop = _preprocessBox(image, box)
    pred = model.predict(crop)
    text = _decodePrediction(pred)

    # If text exists, append to results
    if text:
      results.append((box, text))
  
  return results