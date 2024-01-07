import datetime

LABELS = ["NP", "CP", "A", "FC", "AC"]  # functional
# LABELS = ["P", "A", "C"] #structural
NUM_LABELS = len(LABELS)
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {val: key for key, val in ID2LABEL.items()}
# -------------------------------------------

NUM_FOLDS = 5
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1

# -------------------------------------------
DEVICE = 'cuda'
CURRENT_DATE = datetime.datetime.now()
