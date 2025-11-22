import kagglehub
import zipfile
import os


def downloadAndSaveKaggleDataset(extract_path: str, kaggle_path: str) -> bool:
  """
  Downloads the dataset to the given path.
  NOTE: 'extract_path' must be nonexistent or empty
  
  Parameters:
    extract_path: Path to the directory to extract the Kaggle Dataset to
    kaggle_path: Path to the kaggle repository to download from
  
  Returns:
    bool: True/False of success.
  """

  # Make sure the doesnt exist.
  if os.path.exists(extract_path) and os.listdir(extract_path):
    return False
  
  print("""
  ----------------------------------------------
  ------------ Proceeding with Setup -----------
  ----------------------------------------------\n""")
  
  # Download dataset
  path = kagglehub.dataset_download(kaggle_path)
  print("Downloaded dataset path:", path)
  
  os.makedirs(extract_path, exist_ok=False)
  
  # Extract all files and folders
  with zipfile.ZipFile(path, 'r') as zip_ref:
      zip_ref.extractall(extract_path)

  print(f"Extracted dataset to {extract_path}")
  return True
