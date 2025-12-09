"""
Text Image Generator Script

Generates synthetic text images from a list of words using TRDG (TextRecognitionDataGenerator).
Designed to create OCR datasets with variations including clean, noisy, and distorted images.

Behavior:
- Creates 'test' and 'validation' sets for each word.
- Stores images in subdirectories named after words.
- Filenames follow the format: sample_0.png, sample_1.png, ...
- Skips words if the folder already exists.
"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # In case TensorFlow is installed, clean up the output

import re
import time
from typing import Final
from trdg.generators import (
  GeneratorFromStrings
)


# Id counter for each word
word_ids: dict = {} # { label: id }


# ********************** beg Functions ********************** #

def _generateBatch(config: dict, words: list[str], output_dir: str) -> int:
  """
  Generates and saves a batch of synthetic text images for given words.

  Uses TRDG with the provided batch configuration to apply augmentations
  (skew, blur, distortion, backgrounds) and stores images in subfolders
  named after each word. Returns the number of images saved.
  
  Parameters
  ----------
    config : Config for the GeneratorFromStrings() class
    words : List of strings to generate text for
    output_dir : directory to output under (subdirectories will be \
                 created within this directory for each word).

  Returns
  -------
    int : Total number of images generated
  """
  
  # ---------------- Constants ---------------
  
  # TRDG vars
  IMG_HEIGHT: Final = 32
  IMG_WIDTH: Final = 256
  TEXT_COLOR: Final = "#282828"
  FONT_PATHS: Final = []
  
  # ---------- Create the generator ----------
  
  print(f"\n--- Running Batch: {config['name']} ({config['count']} variations per word) ---")
  start = time.time()
  
  # Initialize the generator with batch-specific parameters
  generator = GeneratorFromStrings(
    # Core Settings
    count=config["count"],         
    strings=words,                        
    size=IMG_HEIGHT,
    width=IMG_WIDTH,
    fit=True,
    text_color=TEXT_COLOR,
    fonts=FONT_PATHS,
    
    # Augmentation Settings (pulled from the config dictionary)
    skewing_angle=config.get("skewing_angle", 0),
    random_skew=config.get("random_skew", False),
    blur=config.get("blur", 0),
    random_blur=config.get("random_blur", False),
    background_type=config.get("background_type", 0),
    distorsion_type=config.get("distorsion_type", 0),
    distorsion_orientation=config.get("distorsion_orientation", 0),
  )

  # Saving vars
  saved_count = 0
  prev_lbl = None
  save_dir = None
  
  
  def cleanText(text: str) -> str:
    """
    Keep letters, numbers, hyphens, underscores, and dots.
    
    Parameters
    ----------
      text: Text to clean

    Returns
    -------
      str: cleaned text
      
    -------
    """
    cleaned = re.sub(r'[^A-Za-z0-9._-]', '_', text)
    return cleaned.strip('_')
    
  # Loop through the generator and save the images
  for img, lbl in generator:
    # Populate word_ids
    key = cleanText(lbl)
    if key not in word_ids:
      word_ids[key] = 1
    
    # Choose the filename
    if key != prev_lbl:
      # Create new save dir path
      save_dir = os.path.join(output_dir, key)
      
      # Only create the folder if it doesnt exist
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      prev_lbl = key
    
    # Create filename
    filename = f"sample_{word_ids[key]}.png"
    word_ids[key] += 1
    img_path = os.path.join(save_dir, filename)
    
    try:
      img.save(img_path)
      saved_count += 1
    except Exception as e:
      print(f"Error saving image for '{lbl}': {e}")
    
    #print(f"Saving {img_path}")
          
  end = time.time()
  elapsed = end - start
  print(f"Batch '{config['name']}' completed. Saved {saved_count} images. Elapsed: {elapsed:.4f} s")
  return saved_count


def generateDataset(dict_path: str, output_dir: str, images_per_word: int) -> bool:
  """
  Validates input/output paths, reads words from a dictionary
  file, and runs multiple TRDG batches for test and validation sets.
  
  Parameters
  ----------
    dict_path : Path to the dictionary (MUST be in dicts/)
    output_dir : Path to the output directory (WILL output in data/)
    images_per_word : Number of images to generate per word

  Returns
  -------
    bool : Success value
  """
  
  clean_count = int(images_per_word * 0.30)
  noise_count = int(images_per_word * 0.40)
  distortion_count = images_per_word - (clean_count + noise_count)
  
  # If rounding pushed distortion_count negative, set
  # to 30%, even if there is no longer the proper total
  if distortion_count < 0:
    distortion_count = int(images_per_word * 0.30)

  # ---------------- Constants ---------------

  # Dirnames
  DICT_DIRECTORY_PATH: Final = "dicts/"
  DATA_DIRECTORY_PATH: Final = "data/"
  TRAINING_DIRNAME: Final = "train/"
  VALIDATION_DIRNAME: Final = "val/"
  
  # --- Batch Configurations (Crucial for Variation) ---
  # Sum of "count" must equal IMAGES_PER_word (100 in this example)
  # This list ensures diversity across the 100 generations per word.
  BATCH_CONFIGS: Final = [
    # 1. Clean Baseline (30% of data)
    {
      "name": "Clean_Batch",
      "count": clean_count,
      "random_blur": False,
      "random_skew": False, 
      "background_type": 0,
      "distorsion_type": 0
    },
    
    # 2. Noise & Randomness (40% of data)
    {
      "name": "Noise_Random_Batch",
      "count": noise_count,
      "random_blur": True,
      "random_skew": True, 
      "background_type": 1,
      "distorsion_type": 0,
      "blur": 1,
      "skewing_angle": 5
    },
    
    # 3. Distortion & Patterns (30% of data)
    {
      "name": "Distortion_Pattern_Batch",
      "count": distortion_count,
      "random_blur": False,
      "random_skew": False, 
      "background_type": 2,
      "distorsion_type": 2,
      "distorsion_orientation": 0
    }
  ]

  # ------------- Ensure proper names -------------

  # Ensure the paths start with the proper prefix
  if not dict_path.startswith(DICT_DIRECTORY_PATH):
    print(f"Renaming {dict_path}", end='')
    dict_path = os.path.join(DICT_DIRECTORY_PATH, dict_path)
    print(f" to {dict_path}...")
  if not output_dir.startswith(DATA_DIRECTORY_PATH):
    print(f"Renaming {output_dir}", end='')
    output_dir = os.path.join(DATA_DIRECTORY_PATH, output_dir)
    print(f" to {output_dir}...")

  # Checks if the dictionary file exists
  if not os.path.exists(dict_path):
    print(f"Input file [{dict_path}] does not exist.")
    return False

  # Checks if the output directory is valid (empty or non-existent)
  if os.path.exists(output_dir) and bool(os.listdir(output_dir)):
    print(f"Output directory [{output_dir}] already exists with files, please choose a non-existent or empty directory.")
    return False

  # ------------- Gather words and run trdg -------------
  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory set to: {output_dir}")
  
  # Split each word in a word up into it's own slot
  print("Reading dictionary...")
  with open(dict_path, "r") as f:
    words = [word for line in f if line.strip() for word in line.strip().split()]
  print("Dictionary read.!")
  
  os.makedirs(os.path.join(output_dir, TRAINING_DIRNAME), exist_ok=False)
  os.makedirs(os.path.join(output_dir, VALIDATION_DIRNAME), exist_ok=False)
  
  start_time = time.time()
  print("\n******************************************", end='')
  print("\n****** Generating training dataset: ******", end='')
  print("\n******************************************")
  for word in words:
    for config in BATCH_CONFIGS:
      _generateBatch(config, [word], os.path.join(output_dir, TRAINING_DIRNAME))
  
  print("\n********************************************", end='')
  print("\n****** Generating validation dataset: ******", end='')
  print("\n********************************************")
  word_ids.clear() # Reset word_ids
  for word in words:
    for config in BATCH_CONFIGS:
      _generateBatch(config, [word], os.path.join(output_dir, VALIDATION_DIRNAME))
    
  end_time = time.time()
  elapsed = end_time - start_time
  print(f"\nTotal time elapsed: {elapsed:.4f} s\n")
  
  return True

# ********************** end Functions ********************** #
