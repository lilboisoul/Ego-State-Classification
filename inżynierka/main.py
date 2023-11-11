import numpy as np
import pandas as pd
import torch.utils.data
import config
from config import *
from helper import *
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

_train = False
_evaluate = False
_prompt = False

if (_train == True or _evaluate == True):
    # df = pd.read_csv("data/goemotions-full-label.csv", delimiter=";")
    df = load_dataset("data/goemotions-full-label.csv")
    print("Before downsampling: ")
    print(df['label'].value_counts())
    df = downsample_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2,
                                                        stratify=df["label"])
    y_train = y_train.to_numpy()  # KeyError fix
    y_test = y_test.to_numpy()  # KeyError fix

    # X_train_list = list(X_train)
    # X_test_list = list(X_test)
    X_train_tokenized = config.neural_net.tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
    X_test_tokenized = config.neural_net.tokenizer(list(X_test), padding=True, truncation=True, max_length=512)

    train_dataset = CustomDataset(X_train_tokenized, y_train)
    test_dataset = CustomDataset(X_test_tokenized, y_test)
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

    args = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    if _train:
        print(trainer.train())
    if _evaluate:
        print(trainer.evaluate())

if _prompt == True:
    np.set_printoptions(suppress=True)

    text = "I need to see a doctor"
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to('cuda')
    outputs = model(**inputs)
    print(outputs)  # before softmax
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # print(predictions) #after softmax
    predictions = predictions.cpu().detach().numpy()
    count = 0
    labels = ["NP", "CP", "A", "FC", "AC"]
    for pred in predictions:
        print(pred * 100.0)
        count += pred  # prettier form
    print(count)
