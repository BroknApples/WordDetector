"""
Generate synthetic text images from a list of phrases.

Usage:
  python scripts/gen_images.py <dict_path.txt> <output_dirname> <images_per_phrase>

Arguments:
  dict_path.txt       File containing phrases, one per line.
  output_dirname      Name of the dataset folder to create under `data/`.
  images_per_phrase   Number of images to generate per phrase.

Description:
  Reads phrases from the given file and generates a dataset of text images.
"""

import sys
import os

# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.generate_dataset import (
  generateDataset
)

def main(argc: int, argv: list[str]) -> int:
  """
  Entry point for generating text images from a phrase list.

  Parses command-line arguments, validates input/output paths, reads
  phrases from a dictionary file, and runs multiple TRDG batches for
  test and validation sets.

  Parameters
  ----------
    argc : Number of command-line arguments provided.
    argv : List of command-line arguments. Expected order:
             
             0: script name
             
             1: path to dictionary file containing phrases
             
             2: name of the output directory (inside 'data/')

             3: number of images to generate per phrase

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
    
  -------
  """

  # Bounds checks
  if argc != 4:
    print(f"Usage: {argv[0]} <dict_path.txt> <output_dirname> <count>")
    return -1
  
  # Get command line args and check paths
  dict_path = argv[1]
  output_dir = argv[2]
  images_per_phrase = int(argv[3])
  
  if not generateDataset(dict_path, output_dir, images_per_phrase):
    return -1
  
  return 0


# Run automatically if ran as the main script.
if __name__ == "__main__":
  print("")
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
