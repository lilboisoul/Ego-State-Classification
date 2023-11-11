import pandas as pd
import torch
import torch.utils.data
import numpy as np

import config.globals
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


def load_dataset(dataset_location):
    df = pd.read_csv(dataset_location, delimiter=";")
    return df


def downsample_dataset(dataset):
    lowest_shape = float('inf')
    for label in config.globals.id2label.keys():
        df_label = dataset[dataset['label'] == label]
        lowest_shape = min(lowest_shape, df_label.shape[0])

    downsampled_dfs = []
    for label in config.globals.id2label.keys():
        df_label = dataset[dataset['label'] == label]
        downsampled_df = df_label.sample(lowest_shape)
        downsampled_dfs.append(downsampled_df)

    df_balanced = pd.concat(downsampled_dfs)
    print(df_balanced['label'].value_counts())
    return df_balanced
