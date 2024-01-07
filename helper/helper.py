import os
import datetime
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from config import *
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def compute_metrics_evaluation(p):
    p_pred, p_labels = p
    p_pred = np.argmax(p_pred, axis=1)
    accuracy = accuracy_score(y_true=p_labels, y_pred=p_pred)
    recall = recall_score(y_true=p_labels, y_pred=p_pred, average='macro')
    precision = precision_score(y_true=p_labels, y_pred=p_pred, average='macro')
    f1 = f1_score(y_true=p_labels, y_pred=p_pred, average='macro')

    conf_matrix = confusion_matrix(p_labels, p_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
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


def compute_metrics_training(p):
    p_pred, p_labels = p
    p_pred = np.argmax(p_pred, axis=1)
    accuracy = accuracy_score(y_true=p_labels, y_pred=p_pred)
    recall = recall_score(y_true=p_labels, y_pred=p_pred, average='macro')
    precision = precision_score(y_true=p_labels, y_pred=p_pred, average='macro')
    f1 = f1_score(y_true=p_labels, y_pred=p_pred, average='macro')

    conf_matrix = confusion_matrix(p_labels, p_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
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


def current_date():
    current_datetime = datetime.datetime.now()
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


def hyperparameter_train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        model = bert_model.to('cuda')
    else:
        model = bert_model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    df = pd.read_csv(r"/data/small_dataset.csv", delimiter=";")

    X = df["text"]
    y = df["label"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train_tokenized = tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(list(X_val), padding=True, truncation=True, max_length=512)

    train_dataset = CustomDataset(X_train_tokenized, y_train)
    val_dataset = CustomDataset(X_val_tokenized, y_val)

    search_space = {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "warmup_ratio": [0.1, 0.2, 0.3],
        "batch_size": [8, 16, 24],
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
                    num_train_epochs=5,
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

                eval_results = trainer.evaluate()
                val_loss = eval_results.get("eval_loss")
                results.append({
                    "learning_rate": learning_rate,
                    "warmup_ratio": warmup_ratio,
                    "batch_size": batch_size,
                    "val_loss": val_loss,
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter-search-results-small", index=False)
