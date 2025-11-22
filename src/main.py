import os
from utilities import (
  downloadAndSaveKaggleDataset
)


def main() -> int:
  """
  Entry point for the main program.
  
  Returns:
    int: How the operation proceeded
  """

  # Download the dataset.
  KAGGLE_DATASET_PATH: str = "vaibhao/handwritten-characters"
  DATASET_SAVE_PATH: str = "../data"
  if not downloadAndSaveKaggleDataset(DATASET_SAVE_PATH, KAGGLE_DATASET_PATH):
    print(f"Error: Could not acuire Kaggle dataset with params: ({DATASET_SAVE_PATH}, {KAGGLE_DATASET_PATH})")
    return -1
  
  
  return 0


# Run automatically if ran as the main script.
if __name__ == "__main__":
  ret: int = main()