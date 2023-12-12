import datetime
import os

import pandas as pd
import torch.cuda
from transformers import BertTokenizer

# -----------------LABELS--------------------
#labels = ["NP", "CP", "A", "FC", "AC"] #functional
labels = ["P", "A", "C"] #structural
num_labels = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
label2id = {val: key for key, val in id2label.items()}
current_date = datetime.datetime.now()
# -------------------------------------------
NUM_FOLDS = 5
NUM_EPOCHS = 4
BATCH_SIZE = 16
#BASE_MODEL = 'bert-base-cased' TO CHECK
#TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
BASE_MODEL = 'bert-base-uncased'
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

#CURRENT_MODEL = Bert
#TRAIN_DATASET =
#TEST_DATASET =
#VAL_DATASET =

DEVICE = 'cuda'

def choose_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"Using: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("Using: cpu")
