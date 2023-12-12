import json
import random

import pandas as pd

from helper import downsample_dataset


def emotions_to_id(emotions_file, result_file):
    with open(emotions_file, 'r') as file:
        emotions = [line.strip() for line in file]

    emotion_map = {
        "NP": ["gratitude", "pride", "admiration", "caring", "approval"],
        "CP": ["disapproval", "disappointment"],
        "A": ["surprise", "realization", "neutral"],
        "FC": ["joy", "love", "amusement", "excitement", "curiosity", "optimism", "desire", "relief"],
        "AC": ["confusion", "anger", "annoyance", "sadness", "embarrassment", "grief", "remorse", "disgust", "fear",
               "nervousness"]
    }

    emotion_to_id = {}
    for emotion_id, emotion_list in emotion_map.items():
        for emotion in emotion_list:
            if emotion in emotions:
                emotion_to_id[emotion] = emotion_id

    emotions_as_ids = [emotion_to_id[emotion] for emotion in emotions]

    print(emotions_as_ids)
    with open(result_file, 'w') as json_file:
        json.dump(emotion_to_id, json_file)


def remap_dataset(dataset, output_file_path):
    functional_remap_dict = {
        0: [0, 4, 5, 15, 21],
        1: [9, 10],
        2: [22, 26, 27],
        3: [1, 7, 8, 13, 17, 18, 20, 23],
        4: [2, 3, 6, 11, 12, 14, 16, 19, 24, 25]
    }

    structural_remap_dict = {
        0: [0, 4, 5, 9, 10, 15, 21],
        1: [22, 26, 27],
        2: [1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 23, 24, 25]
    }

    reverse_map = {v: k for k, values in structural_remap_dict.items() for v in values}

    def process_and_remap(label_str):
        if label_str == ',' or not label_str:
            return None

        label_list = [int(label) for label in label_str.split(',') if label.isdigit()]

        remapped_labels = [reverse_map.get(label, label) for label in label_list]
        return random.choice(remapped_labels) if remapped_labels else None

    dataset['label'] = dataset['label'].apply(process_and_remap)

    dataset = dataset.dropna(subset=['label'])

    dataset['label'] = dataset['label'].astype(int)

    dataset.drop(columns=['id'], inplace=True)

    if not output_file_path:
        output_file_path = 'data/dataset.tsv'
    dataset.to_csv(output_file_path, sep="\t", index=False)

    return dataset


def original_dataset_to_mapped_dataset(original_dataset_path, remapped_dataset_path):
    df = pd.read_csv(original_dataset_path, delimiter=";")
    print(df.head())

    functional_label_mapping = {
        0: ["gratitude", "pride", "admiration", "caring", "approval"],
        1: ["disapproval", "disappointment"],
        2: ["surprise", "realization", "neutral"],
        3: ["joy", "love", "amusement", "excitement", "curiosity", "optimism", "desire", "relief"],
        4: ["confusion", "anger", "annoyance", "sadness", "embarrassment", "grief", "remorse", "disgust", "fear",
            "nervousness"]
    }
    structural_label_mapping = {
        0: ["gratitude", "pride", "admiration", "caring", "approval", "disapproval", "disappointment"],
        1: ["surprise", "realization", "neutral"],
        2: ["joy", "love", "amusement", "excitement", "curiosity", "optimism", "desire", "relief", "confusion", "anger", "annoyance", "sadness", "embarrassment", "grief", "remorse", "disgust", "fear",
            "nervousness"],
    }

    df['label'] = df.iloc[:, 1:].apply(
        lambda row: next((label for label, emotions in structural_label_mapping.items() if any(row[emotions])), None), axis=1)


    data = df.drop(df.columns[1:29], axis=1)
    data.to_csv(remapped_dataset_path, sep=";", index=False)
    print(data.head())


def subdataset_generator():
    df = pd.read_csv("data/dataset-raw.csv", delimiter=";")
    # df = load_dataset("data/goemotions-1000-label.csv")
    df = downsample_dataset(df)

    small_dataset = []
    for label in df['label'].unique():
        samples = df[df['label'] == label].sample(1000, random_state=42)
        # Append the sampled data to the new DataFrame
        small_dataset.append(samples)
    small_dataset = pd.concat(small_dataset)
    small_dataset = small_dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # print(small_dataset['label'].value_counts())
    small_dataset.to_csv(path_or_buf="small_dataset.csv", sep=";", index=False)
