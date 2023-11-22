import pandas as pd
import torch
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

from config import *
from transformers import Trainer, TrainingArguments

import config.globals
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix


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
    # print(type(p))
    p_pred, p_labels = p
    p_pred = np.argmax(p_pred, axis=1)
    accuracy = accuracy_score(y_true=p_labels, y_pred=p_pred)
    recall = recall_score(y_true=p_labels, y_pred=p_pred, average='macro')
    precision = precision_score(y_true=p_labels, y_pred=p_pred, average='macro')
    f1 = f1_score(y_true=p_labels, y_pred=p_pred, average='macro')

    # conf_matrix = confusion_matrix(p_labels, p_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    # plt.title("Confusion Matrix")
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.show()

    return {"accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1}


def load_dataset(dataset_location):
    df = pd.read_csv(dataset_location, delimiter=";")
    return df


def downsample_dataset(dataset):
    print("Before downsampling: ")
    print(dataset['label'].value_counts())
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


def train_with_crossvalidation(model, tokenizer, X, y):
    print(f"Training model: {model.__class__.__name__}")
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_tokenized = tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
        X_val_tokenized = tokenizer(list(X_val), padding=True, truncation=True, max_length=512)

        train_dataset = CustomDataset(X_train_tokenized, y_train)
        eval_dataset = CustomDataset(X_val_tokenized, y_val)

        args = TrainingArguments(
            output_dir=f"output-{current_date()}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        # if _train:
        print(f"Training fold {fold + 1}...")
        trainer.train()
        # print(trainer.train())
        # if _evaluate:
        print(trainer.evaluate())
        # trainer.evaluate()


def current_date():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%d-%m-%Y-%H-%M")

    return formatted_datetime


def get_predictions(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to('cuda')
    outputs = model(**inputs)
    # print(outputs)  # before softmax
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # print(predictions) #after softmax
    predictions = predictions.cpu().detach().numpy()
    return predictions


def format_and_print_predictions(predictions):
    labels = config.globals.labels
    predictions = predictions[0]
    for label, pred in zip(labels, predictions):
        print(f"{label} - {float(pred):.3f}")


def evaluate_model_on_test_set(tokenizer, X_test, y_test, evaluated_model):
    with open(f'eval-{current_date()}.txt', 'w') as file:
        X_test_tokenized = tokenizer(list(X_test), padding=True, truncation=True, max_length=512)
        test_dataset = CustomDataset(X_test_tokenized, y_test)

        trainer = Trainer(
            model=evaluated_model,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        eval_result = trainer.evaluate()
        for key, value in eval_result.items():
            file.write(f"{key} -> {value}\n")
