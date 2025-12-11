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


import numpy as np
import cv2
from typing import Final


# ---------------- Constants ---------------

# Load network
EAST_MODEL: Final = "frozen_east_text_detection.pb"
EAST = cv2.dnn.readNet(EAST_MODEL)

# Output layer names from EAST
LAYERS: Final = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# Mean values used in blobbing for the EAST model
EAST_MEANS: Final = (123.68, 116.78, 103.94)

# Blob scale factor
BLOB_SCALE_FACTOR: Final = 1.0

# Required size for use by the EAST model; each tier is more accurate, but slower.
REQ_SIZES: Final = (
  (320, 320), # Fast and accurate
  (512, 512), # Slower, but more accurate
  (640, 640)  # Slow, but very accurate
)

NMS_SCORE_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.4


# ********************** beg Functions ********************** #

def _decode(scores: np.ndarray, geometry: np.ndarray, thresh: float = 0.5) -> tuple[list[tuple[float, float, float, float]], list[float]]:
  """
  Decode EAST text detector predictions.

  Parameters
  ----------
    scores : Probability map from EAST model, shape (1, 1, H/4, W/4)
    geometry : Bounding box geometry from EAST model, shape (1, 5, H/4, W/4)
    thresh : Minimum confidence threshold to accept a box (default=0.5)

  Returns
  -------
    rects : List of bounding boxes as (start_x, start_y, end_x, end_y)
    confidences : List of confidence scores corresponding to each box

  -------
  """
  
  rows, cols = scores.shape[2:4]
  rects = []
  confidences = []

  for y in range(rows):
    data = scores[0, 0, y]
    x0 = geometry[0, 0, y]
    x1 = geometry[0, 1, y]
    x2 = geometry[0, 2, y]
    x3 = geometry[0, 3, y]
    angles = geometry[0, 4, y]

    for x in range(cols):
      if data[x] < thresh: continue
      score = data[x]

      angle = angles[x]
      cos = np.cos(angle)
      sin = np.sin(angle)

      h = x0[x] + x2[x]
      w = x1[x] + x3[x]

      offset_x = x * 4.0
      offset_y = y * 4.0

      end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
      end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
      start_x = int(end_x - w)
      start_y = int(end_y - h)

      # Clamp
      start_x = max(0, start_x)
      start_y = max(0, start_y)
      end_x = max(0, end_x)
      end_y = max(0, end_y)

      rects.append((start_x, start_y, end_x, end_y))
      confidences.append(float(score))

  return rects, confidences


def detectTextBBFromImage(image: np.ndarray, size_type: int = 0, show_img: bool = False, box_color: tuple[int, int, int] = (0, 255, 0), box_thickness: int = 2) -> list[tuple[int, int, int, int]]:
  """
  Gets the bounding box locations for likely 
  text-regions in an image using the EAST model.

  Parameters
  ----------
    image: Numpy array for your image
    size_type : Size of the resized images -> 0 = (320, 320), 1 = (512, 512), 2 = (640, 640)
    show_img : Should the image be displayed with bounding boxes before returning?
    box_color : Color to render the boxes as on the image if show_img is set to true
    box_thickness: Thickness of the box outlines to render on the image if show-Img is set to true

  Returns
  -------
    boxes : Bounding box locations formatted as (start_x, start_y, end_x, end_y)
    
  -------
  """

  # Bounds checks
  if size_type < 0 or size_type > 2:
    raise ValueError("size_type must be (0, 1, or 2)")
  elif box_thickness == 0 or box_thickness < -1:
    raise ValueError("box_thickness must be -1 or greater than 0")

  # ------------- Run EAST model -------------

  # Copy and resize image
  orig = image.copy() # Copy of the original image used for mapping bounding boxes
  image = cv2.resize(image, REQ_SIZES[size_type])

  # Create blob
  blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=BLOB_SCALE_FACTOR,
    size=REQ_SIZES[size_type],
    mean=EAST_MEANS,
    swapRB=True,
    crop=False
  )

  # Forward pass
  EAST.setInput(blob)
  scores, geometry = EAST.forward(LAYERS)

  # Decode into boxes
  rects, confidences = _decode(scores, geometry)

  # Apply non-max suppression
  rects_xywh = []
  for (x1, y1, x2, y2) in rects:
    rects_xywh.append([x1, y1, x2 - x1, y2 - y1])
  boxes = cv2.dnn.NMSBoxes(rects_xywh, confidences, NMS_SCORE_THRESHOLD, NMS_THRESHOLD)

  # Safety check
  if len(boxes) == 0: return []

  # Scale back to original image
  orig_h, orig_w = orig.shape[:2]
  scale_w = orig_w / REQ_SIZES[size_type][0]
  scale_h = orig_h / REQ_SIZES[size_type][1]
  scaled_boxes = []
  for i in boxes.flatten():
    start_x, start_y, end_x, end_y = rects[i]
    start_x = int(start_x * scale_w)
    start_y = int(start_y * scale_h)
    end_x   = int(end_x * scale_w)
    end_y   = int(end_y * scale_h)
    scaled_boxes.append((start_x, start_y, end_x, end_y))

  # Draw result
  if show_img:
    for start_x, start_y, end_x, end_y in scaled_boxes:
      cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), box_color, box_thickness)
    
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
  
  return scaled_boxes


def detectTextBBFromImages(images: list[np.ndarray], size_type: int = 0, show_img: bool = False, box_color: tuple[int, int, int] = (0, 255, 0), box_thickness: int = 2) -> list[list[tuple[int, int, int, int]]]:
  """
  Gets the bounding box locations for likely 
  text-regions in an image using the EAST model.

  Parameters
  ----------
    image : List of loaded images
    size_type : Size of the resized images -> 0 = (320, 320), 1 = (512, 512), 2 = (640, 640)
    show_img : Should the image be displayed with bounding boxes before returning?
    box_color : Color to render the boxes as on the image if show_img is set to true
    box_thickness: Thickness of the box outlines to render on the image if show-Img is set to true

  Returns
  -------
    boxes : List of bounding box locations formatted as [(start_x1, start_y1, end_x1, end_y1), (start_x2, start_y2, end_x2, end_y2), ...]
    
  -------
  """
  boxes = []
  for image in images:
    boxes.append(detectTextBBFromImage(image, size_type, show_img, box_color, box_thickness))
  
  return boxes

# ********************** end Functions ********************** #
