"""
CRNN + CTC Training on Synthetic Word Images
"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import cv2
from typing import Final

# -------------------- Constants --------------------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256
CHARS: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"
MAX_LABEL_LEN: Final = 16 # max characters per word
NUM_CLASSES: Final = len(CHARS) + 1 # +1 for CTC blank
MODELS_DIR: Final = "models/"
BATCH_SIZE: Final = 32
NUM_EPOCHS: Final = 150

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}


# GPU detection
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  print(f"TensorFlow found the GPU(s): {gpus}")
  # Optional: Set memory growth to prevent the GPU from grabbing all VRAM at once
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
else:
  print("TensorFlow did not detect any GPU.")


# -------------------- Dataset Preparation --------------------
def encodeWord(word: str) -> np.ndarray:
  """
  Convert word to integer sequence and pad to MAX_LABEL_LEN
  """
  # Removed .lower() call so the model learns 'A' vs 'a'
  arr = [CHAR_TO_IDX.get(c, 0) for c in word] 
  if len(arr) < MAX_LABEL_LEN:
    arr += [NUM_CLASSES-1] * (MAX_LABEL_LEN - len(arr)) # pad with CTC blank
  else:
    arr = arr[:MAX_LABEL_LEN]
  return np.array(arr, dtype=np.int32)


def prepareDataset(data_dir: str):
  """
  Loads all images and labels from synthetic dataset directory.

  Parameters
  ----------
    data_dir: Directory for data
  
  Returns
  -------
    Dataset usuable on the crnn
  
  -------
  """

  images, labels, input_lens, label_lens = [], [], [], []

  for word_dir in os.listdir(data_dir):
    word_path = os.path.join(data_dir, word_dir)
    if not os.path.isdir(word_path): continue

    # NOTE: Using the directory name as the label!
    word_label = word_dir
    if not word_label: continue
        
    for img_file in os.listdir(word_path):
      img_path = os.path.join(word_path, img_file)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      if img is None: continue

      # Preprocessing (must match inference)
      img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
      img = img.astype("float32") / 255.0
      img = np.expand_dims(img, axis=-1)
      images.append(img)

      label_seq = encodeWord(word_label)
      labels.append(label_seq)

      # Input length after CNN (2 maxpools of stride 2) = IMG_WIDTH / 4
      input_lens.append([IMG_WIDTH // 4])
      label_lens.append([min(len(word_label), MAX_LABEL_LEN)])

  return {
    "images": np.array(images, dtype=np.float32),
    "labels": np.array(labels, dtype=np.int32),
    "input_lens": np.array(input_lens, dtype=np.int32),
    "label_lens": np.array(label_lens, dtype=np.int32)
  }


def buildCrnnModel():
  """
  Builds a crnn model from scratch

  Returns
  -------
    Training model used for creating weights
    Prediction model used for actual use

  -------
  """
  
  input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
  labels = keras.Input(shape=(MAX_LABEL_LEN,), dtype="int32", name="label")
  input_len = keras.Input(shape=(1,), dtype="int32", name="input_len")
  label_len = keras.Input(shape=(1,), dtype="int32", name="label_len")

  # Conv layers
  x = layers.Conv2D(64, 3, padding="same", activation="relu")(input_img)
  x = layers.MaxPooling2D((2,2))(x)
  x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
  x = layers.MaxPooling2D((2,2))(x)

  # Collapse height (pool_h * filters)
  pool_h, pool_w = x.shape[1], x.shape[2]
  x = layers.Reshape(target_shape=(pool_w, pool_h * 128))(x)

  # Bidirectional LSTM
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

  # Output -- Softmax over all characters + blank
  y_pred = layers.Dense(NUM_CLASSES, activation="softmax", name="y_pred")(x)

  # 1. Prediction Model (The one for inference)
  prediction_model = keras.Model(
    inputs=input_img,
    outputs=y_pred,
    name='crnn_predictor'
  )

  # 2. Training Model (For CTC Loss and fitting)
  def ctcLambda(args):
    y_pred, labels, input_len, label_len = args
    # CTC expects (Time_steps, Batch, Classes), but y_pred is (Batch, Time_steps, Classes)
    return keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)

  loss_out = layers.Lambda(ctcLambda, output_shape=(1,), name="ctc")(
    [y_pred, labels, input_len, label_len]
  )

  training_model = keras.Model(
    inputs=[input_img, labels, input_len, label_len],
    outputs=loss_out
  )
  training_model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred) # Dummy loss

  return {
    "training": training_model,
    "prediction": prediction_model
  }

# -------------------- Training --------------------
def trainCustomCrnn(base_data_dir: str, model_name: str, plot_history: bool = False):
  """
  Trains the custom crnn for use

  Parameters
  ----------
    base_data_dir : Base data directory
    model_name : Name of the model
    plot_history : Show the history of the plot when finished?
  
  -------
  """
  
  # Load Training and Validation Data
  train_dir = os.path.join(base_data_dir, "train")
  val_dir = os.path.join(base_data_dir, "val")
  
  if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    raise FileNotFoundError(f"Error: Ensure 'train/' and 'val/' directories exist under {base_data_dir}")

  train_dataset = prepareDataset(train_dir)
  val_dataset = prepareDataset(val_dir)
  print(f"Loaded {len(train_dataset['images'])} training samples.")
  print(f"Loaded {len(val_dataset['images'])} validation samples.")
  
  # Build Models
  models = buildCrnnModel()
  training_model = models['training']
  prediction_model = models['prediction'] # The model we will save at the end

  # Paths
  os.makedirs(MODELS_DIR, exist_ok=True)
  checkpoint_weights_path = os.path.join(MODELS_DIR, (model_name + ".weights.h5"))
  final_predictor_path = os.path.join(MODELS_DIR, (model_name + "_predictor.keras"))

  if os.path.exists(checkpoint_weights_path):
    print(f"\nResuming training: Loading existing weights from {checkpoint_weights_path}...")
    try:
      training_model.load_weights(checkpoint_weights_path)
    except Exception as e:
      print(f"Error loading weights: {e}. Starting fresh.")

  # Callbacks
  early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
  checkpoint = ModelCheckpoint(
    checkpoint_weights_path, 
    monitor="val_loss", 
    save_weights_only=True, # Saves only the weights of the *training* model
    save_best_only=True, 
    verbose=1
  )

  # Prepare inputs/outputs (outputs are dummies for CTC)
  train_inputs = [train_dataset["images"], train_dataset["labels"], train_dataset["input_lens"], train_dataset["label_lens"]]
  train_outputs = np.zeros((len(train_dataset["images"]), 1)) 

  val_inputs = [val_dataset["images"], val_dataset["labels"], val_dataset["input_lens"], val_dataset["label_lens"]]
  val_outputs = np.zeros((len(val_dataset["images"]), 1))

  # Training (This is your 15-minute step)
  print("Starting training...")
  history = training_model.fit(
    x=train_inputs,
    y=train_outputs,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(val_inputs, val_outputs),
    callbacks=[early_stop, checkpoint]
  )
  
  # Final Save
  # The training model was restored to its best weights by EarlyStopping.
  # Load those best weights from the checkpoint file into the prediction model structure.
  print(f"Loading best weights from {checkpoint_weights_path}...")
  prediction_model.load_weights(checkpoint_weights_path)
  
  # Save the final, usable prediction model
  prediction_model.save(final_predictor_path)
  print(f"Training complete. FINAL PREDICTOR MODEL saved as {final_predictor_path}")

  if plot_history:
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CRNN Training Loss")
    plt.legend()
    plt.show()
    
  return True
