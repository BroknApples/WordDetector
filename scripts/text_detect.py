"""
Usage: python scripts/text_detect.py <image_path> <model_path> [resize] [display_bb]
"""

import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from typing import Final
from tensorflow.keras.models import load_model # type: ignore

from src.east_text_detection import detectTextBBFromImage, detectTextBBFromImages
from src.custom_crnn_text_recognition import recognizeTextFromBBs_CustomCRNN
from src.easyocr_text_recognition import recognizeTextFromBBs_EasyOCR


CRNN_MODEL_TYPE: Final = "crnn"
EASYOCR_MODEL_TYPE: Final = "easyocr"
MIN_OCR_CONFIDENCE: Final = 0.5
FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: Final = 0.7
FONT_COLOR: Final = (255, 0, 0)
FONT_THICKNESS: Final = 2
CUSTOM_CRNN_MODEL_PATH: Final = "models/crnn_predictor.keras"
IMAGE_EXTENSIONS: Final = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'] # Common image extensions

def _sortBoxesReadingOrder(boxes, y_thresh: int = 12):
  """
  Sorts boxes by their reading order (top-to-bottom, left-to-right)

  Parameters
  ----------
    boxes : List of bounding-box coordinates
    y_thresh : Threshold for boxes to be considered on the same line

  Returns
  -------
    result: List of results
  
  -------
  """
  
  # group into lines by y coordinate, then sort within line by x
  if not boxes:
    return []
  
  boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
  lines = []
  current_line = [boxes_sorted[0]]
  for b in boxes_sorted[1:]:
    if abs(b[1] - current_line[0][1]) <= y_thresh:
      current_line.append(b)
    else:
      lines.append(sorted(current_line, key=lambda bb: bb[0]))
      current_line = [b]
  lines.append(sorted(current_line, key=lambda bb: bb[0]))
  
  # Flatten list of lists back into a single list
  res = [item for sublist in lines for item in sublist]
  return res


# ---------- Main ----------
def main(argc: int, argv: list[str]) -> int:
  """
  Entry point for generating text images from a phrase list.

  Runs the text detector pipeline

  Parameters
  ----------
    argc : Number of command-line arguments provided.
    argv : List of command-line arguments. Expected order:
             
             0: script name
             
             1: Name of the file or directory of files to analyze

             3: model to use ("custom" or "easyocr")

             3: Resize parameter for the EAST model

             4: display results once completed

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
  
  -------
  """

  if argc < 3 or argc > 5:
    print("Usage: python text_detect.py <image_path> <model_type> [resize] [display]")
    return -1

  images_path = argv[1]
  model_type = argv[2].lower()
  resize = int(argv[3]) if len(argv) > 3 else 1
  display = (argv[4].lower() in ("1", "true", "yes")) if len(argv) > 4 else False

  if not os.path.exists(images_path):
    print(f"Invalid image path [{images_path}]")
    return -1

  if model_type != CRNN_MODEL_TYPE and model_type != EASYOCR_MODEL_TYPE:
    print(f"Invalid model_type [{model_type}]")
    return -1

  # Ensure output dir exists
  OUTPUT_DIR = Path("out")
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  # Determine list of image files to process
  image_paths: list[str] = []
  
  if os.path.isdir(images_path):
    print(f"Processing directory: {images_path}")
    # Loop through all files in the directory
    for filename in os.listdir(images_path):
      if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        image_paths.append(os.path.join(images_path, filename))
    
    if not image_paths:
      print(f"No supported image files found in {images_path}")
      return -1
  elif os.path.isfile(images_path):
    # Single file case
    image_paths.append(images_path)

  
  if model_type == CRNN_MODEL_TYPE:
    model = load_model(CUSTOM_CRNN_MODEL_PATH)

  def runPipeline(img_path):
    """
    Run the full pipeline on an image

    Parameters
    ----------
      img_path: Path to an image to load
    """
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None: return # Skip unreadable images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect text boxes & sort
    boxes = detectTextBBFromImage(image, resize, False)
    boxes = _sortBoxesReadingOrder(boxes)

    copy = image.copy()

    # CRNN Model pipeline
    if model_type == CRNN_MODEL_TYPE:
      results = recognizeTextFromBBs_CustomCRNN(image, boxes, model)
      for tup in results:
        (x1, y1, x2, y2), word_list = tup
        word_string = word_list[0]
        print(f"{(x1, y1, x2, y2)} : {word}")
        
        # Draw Bounding Box
        cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(copy, word_string, (x1, y1 - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    # EASYOCR Model pipeline
    elif model_type == EASYOCR_MODEL_TYPE:
      results = recognizeTextFromBBs_EasyOCR(image, boxes)
      for (box, word, avg_conf) in results:
        x1, y1, x2, y2 = box
        print(f"{(x1, y1, x2, y2)} : {word}")
        
        # Skip if word is empty or confidence is too low
        if not word or avg_conf < MIN_OCR_CONFIDENCE:
            continue
            
        print(f"[{word}] (Conf: {avg_conf:.2f})")

        cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(copy, word, (x1, y1 - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    
    # Print and draw
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    if display:
      cv2.imshow("Result", copy)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    save_path = OUTPUT_DIR / f"{name}_{append}.png"
    cv2.imwrite(str(save_path), copy)

  # Run pipeline for each image
  for img_path in image_paths:
    runPipeline(img_path)

  return 0


# Run automatically if ran as the main script
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)
  if ret == 0:
    print("Operation completed successfully!")
  else:
    print("Operation did not complete properly. Error detected.")
