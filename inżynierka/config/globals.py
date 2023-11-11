import datetime
import os

import torch.cuda

# -----------------LABELS--------------------
labels = ["NP", "CP", "A", "FC", "AC"]
num_labels = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
label2id = {val: key for key, val in id2label.items()}
# -------------------------------------------

CURRENT_RESULT_PATH = ""
CURRENT_DATASET_PATH = ""
DATA_PATH = r'.\data'
DATASET_FILENAME = r'goemotions-processed.csv'

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("Using: cpu")
