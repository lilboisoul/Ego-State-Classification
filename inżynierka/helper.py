import json
import random

import pandas as pd
import torch
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ray import tune
from sklearn.model_selection import StratifiedKFold, train_test_split

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

    conf_matrix = confusion_matrix(p_labels, p_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plot_file_path = os.path.join("confusion-matrix.png")
    plt.savefig(plot_file_path)
    return {"accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1}


def load_dataset(path, delimiter):
    df = pd.read_csv(path, delimiter=delimiter)
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


def init_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"Using: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("Using: cpu")


def plot_metrics(metrics):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, NUM_FOLDS + 1), metrics['train_loss'], marker='o', color='r')
    plt.title("Loss per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Train Loss")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, NUM_FOLDS + 1), metrics['eval_loss'], marker='o', color='g')
    plt.title("Loss per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Eval Loss")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, NUM_FOLDS + 1), metrics['f1'], marker='o', color='r')
    plt.title("F1 per Fold")
    plt.xlabel("Fold")
    plt.ylabel("F1")

    plt.tight_layout()
    return plt

def hyperparameter_train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        model = bert_model.to('cuda')
    else:
        model = bert_model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    df = pd.read_csv(r"C:\Projekty\inżynierka\data\small_dataset.csv", delimiter=";")

    X = df["text"]
    y = df["label"].to_numpy()

    # Create 85/15 train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train_tokenized = tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(list(X_val), padding=True, truncation=True, max_length=512)

    train_dataset = CustomDataset(X_train_tokenized, y_train)
    val_dataset = CustomDataset(X_val_tokenized, y_val)

    search_space = {
        "learning_rate": [1e-5],
        "warmup_ratio": [0.1, 0.3],
        "batch_size": [8, 16, 24],
        "num_epochs": 5
    }

    results = []
    _labels = ["NP", "CP", "A", "FC", "AC"]
    _num_labels = len(_labels)
    _id2label = {i: label for i, label in enumerate(_labels)}
    _label2id = {val: key for key, val in _id2label.items()}
    for learning_rate in search_space['learning_rate']:
        for warmup_ratio in search_space['warmup_ratio']:
            for batch_size in search_space['batch_size']:
                model = BertForSequenceClassification.from_pretrained(base_model,
                                                                           num_labels=_num_labels,
                                                                           id2label=_id2label,
                                                                           label2id=_label2id)
                if device == "cuda":
                    model = model.to('cuda')
                args = TrainingArguments(
                    output_dir=r"C:\Projekty\inżynierka\test-output-dir",
                    num_train_epochs=search_space["num_epochs"],
                    learning_rate=learning_rate,
                    warmup_ratio=warmup_ratio,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    do_eval=False,
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )

                trainer.train()
                # Compute validation loss
                eval_results = trainer.evaluate()
                val_loss = eval_results.get("eval_loss")
                results.append({
                    "learning_rate": learning_rate,
                    "warmup_ratio": warmup_ratio,
                    "batch_size": batch_size,
                    "val_loss": val_loss,
                })


    # Print or save the results for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter-search-results-small", index=False)


