"""
Text Image Generator Script

Generates synthetic text images from a list of phrases using TRDG (TextRecognitionDataGenerator).
Designed to create OCR datasets with variations including clean, noisy, and distorted images.

Usage:
  python src/gen_images.py <dict_path.txt> <output_dirname> <images_per_phrase>

Arguments:
  dict_path.txt       Text file with phrases/characters, one per line.
  output_dirname      Name of the output dataset directory (created in 'data/').
  images_per_phrase   Number of images to generate per phrase.

Behavior:
- Creates 'test' and 'validation' sets for each phrase.
- Stores images in subdirectories named after phrases.
- Filenames follow the format: sample_0.png, sample_1.png, ...
- Skips phrases if the folder already exists.
"""


import sys
import os
import time
from typing import Final
from trdg.generators import (
  GeneratorFromStrings
)
from utilities import (
  cleanText
)

# Id counter for each phrase
phrase_ids: dict = {} # { label: id }

# ********************** beg Functions ********************** #

def generateBatch(config: dict, phrases: list[str], output_dir: str) -> int:
  """
  Generates and saves a batch of synthetic text images for given phrases.

  Uses TRDG with the provided batch configuration to apply augmentations
  (skew, blur, distortion, backgrounds) and stores images in subfolders
  named after each phrase. Returns the number of images saved.
  
  Parameters
  ----------
    config : Config for the GeneratorFromStrings() class
    phrases : List of strings to generate text for
    output_dir : directory to output under (subdirectories will be \
                 created within this directory for each phrase).

  Returns
  -------
    int : Total number of images generated
  """
  
  # ---------------- Constants ---------------
  
  # TRDG vars
  IMG_HEIGHT: Final = 32
  IMG_WIDTH: Final = -1 # Dynamic width
  TEXT_COLOR: Final = "#282828"
  FONT_PATHS: Final = []
  
  # ---------- Create the generator ----------
  
  print(f"\n--- Running Batch: {config['name']} ({config['count']} variations per phrase) ---")
  start = time.time()
  
  # Initialize the generator with batch-specific parameters
  generator = GeneratorFromStrings(
    # Core Settings
    count=config["count"],         
    strings=phrases,                        
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
  
  # Loop through the generator and save the images
  for img, lbl in generator:
    # Populate phrase_ids
    key = cleanText(lbl)
    if key not in phrase_ids:
      phrase_ids[key] = 1
    
    # Choose the filename
    if key != prev_lbl:
      # Create new save dir path
      save_dir = os.path.join(output_dir, key)
      
      # Only create the folder if it doesnt exist
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      prev_lbl = key
    
    # Create filename
    filename = f"sample_{phrase_ids[key]}.png"
    phrase_ids[key] += 1
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
             
             1: path to dictionary file containing phrases
             
             2: name of the output directory (inside 'data/')

             3: number of images to generate per phrase

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
  """

  # Bounds checks
  if argc != 4:
    print(f"Usage: {argv[0]} <dict_path.txt> <output_dirname> <count>")
    return -1
  
  # Get command line args and check paths
  dict_path = argv[1]
  output_dir = argv[2]
  images_per_phrase = int(argv[3])
  
  clean_count = int(images_per_phrase * 0.30)
  noise_count = int(images_per_phrase * 0.40)
  distortion_count = images_per_phrase - (clean_count + noise_count)
  
  # If rounding pushed distortion_count negative, set
  # to 30%, even if there is no longer the proper total
  if distortion_count < 0:
    distortion_count = int(images_per_phrase * 0.30)

  # ---------------- Constants ---------------

  # Dirnames
  DICT_DIRECTORY_PATH: Final = "dicts/"
  DATA_DIRECTORY_PATH: Final = "data/"
  TEST_DIRNAME: Final = "test/"
  VALIDATION_DIRNAME: Final = "validation/"
  
  # --- Batch Configurations (Crucial for Variation) ---
  # Sum of "count" must equal IMAGES_PER_PHRASE (100 in this example)
  # This list ensures diversity across the 100 generations per phrase.
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
    return -1

  # Checks if the output directory is valid (empty or non-existent)
  if os.path.exists(output_dir) and bool(os.listdir(output_dir)):
    print(f"Output directory [{output_dir}] already exists with files, please choose a non-existent or empty directory.")
    return -1

  # ------------- Gather phrases and run trdg -------------
  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory set to: {output_dir}")
  
  print("Reading dictionary...")
  with open(dict_path, "r") as f:
    phrases = [line.strip() for line in f if line.strip()]
  
  os.makedirs(os.path.join(output_dir, TEST_DIRNAME), exist_ok=False)
  os.makedirs(os.path.join(output_dir, VALIDATION_DIRNAME), exist_ok=False)
  
  start_time = time.time()
  print("\n***************************************", end='')
  print("\n****** Generating test datasets: ******", end='')
  print("\n***************************************")
  for phrase in phrases:
    for config in BATCH_CONFIGS:
      generateBatch(config, [phrase], os.path.join(output_dir, TEST_DIRNAME))
  
  print("\n*********************************************", end='')
  print("\n****** Generating validation datasets: ******", end='')
  print("\n*********************************************")
  phrase_ids.clear() # Reset phrase_ids
  for phrase in phrases:
    for config in BATCH_CONFIGS:
      generateBatch(config, [phrase], os.path.join(output_dir, VALIDATION_DIRNAME))
    
  end_time = time.time()
  elapsed = end_time - start_time
  print(f"\nTotal time elapsed: {elapsed:.4f} s\n")
  
  return 0

# ********************** end Functions ********************** #

# Run automatically if ran as the main script.
if __name__ == "__main__":
  print("")
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
