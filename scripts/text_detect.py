"""


Usage: python src/text_detect.py <image_path>
"""


import sys
import os
import numpy
import cv2
import torch

from east_text_detection import (
  detectTextBBFromImage,
  detectTextBBFromImages,
)


def main(argc: int, argv: list[str]) -> int:
  """
  Entry point for generating synthetic text images from a phrase list.

  Parses command-line arguments, validates input/output paths, reads
  phrases from a dictionary file, and runs multiple TRDG batches for
  test and validation sets.

  Parameters
  ----------
    argc : Number of command-line arguments provided.
    argv : List of command-line arguments. Expected order:
             
             0: script name
             
             1: The image size to be taken by the EAST detector (0 = (320, 320)), (1 = (512, 512)), (2 = (640, 640))

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
  """

  # TODO Should use argv[1] to be the path to the image, argv[2] should be the model path
  
  image_size = int(argv[1])
  
  path = "img1.webp"
  detectTextBBFromImage(path, image_size, True)
  
  return 0


# Run automatically if ran as the main script.
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
