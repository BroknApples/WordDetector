"""

"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import sys
import os
import numpy as np
import easyocr
from typing import Final
# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# CONFIG
RESIZE: Final = 1
DISPLAY: Final = True
MIN_OCR_CONFIDENCE: Final = 0.50

# Initialize EasyOCR Reader once
# Use English ('en') and set GPU=False if you don't have one
reader = easyocr.Reader(['en'], gpu=False) 


def recognizeTextFromBBs_EasyOCR(image, boxes):
  """
  Recognizes text given an image and a list of bounding boxes along with the trained custom crnn model
  
  Parameters
  ----------
    image : Image to read
    boxes : List of bounding-box coordinates on the image

  Returns
  -------
    result: List of results
  
  -------
  """

  results = []

  for box in boxes:
    x1, y1, x2, y2 = box

    # Add some padding
    pad = 2
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(image.shape[1], x2 + pad)
    y2p = min(image.shape[0], y2 + pad)
    crop = image[y1p:y2p, x1p:x2p]

    # Recognition with detail=1 (bbox, text, confidence)
    recognized_segments = reader.readtext(
      crop,
      allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
      detail=1
    )

    word = ""
    confidences = []

    # Sort segments left-to-right to avoid merging issues
    recognized_segments.sort(key=lambda s: s[0][0][0])
    for bbox, text, conf in recognized_segments:
      if conf >= MIN_OCR_CONFIDENCE:
        word += text
        confidences.append(conf)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    results.append((box, word, avg_conf))

  return results

