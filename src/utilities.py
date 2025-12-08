import re


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