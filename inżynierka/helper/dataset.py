import pandas as pd
from config import ID2LABEL
import torch.utils.data
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

def original_dataset_to_mapped_dataset(original_dataset_path, remapped_dataset_path):
    df = pd.read_csv(original_dataset_path, delimiter=";")

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
        2: ["joy", "love", "amusement", "excitement", "curiosity", "optimism", "desire", "relief", "confusion",
            "anger", "annoyance", "sadness", "embarrassment", "grief", "remorse", "disgust", "fear", "nervousness"],
    }
    current_map = structural_label_mapping
    df['label'] = df.iloc[:, 1:].apply(
        lambda row: next((label for label, emotions in current_map.items() if any(row[emotions])), None), axis=1)

    data = df.drop(df.columns[1:29], axis=1)
    data.to_csv(remapped_dataset_path, sep=";", index=False)


def subset_generator(dataset, samples_per_label):
    subset = []
    for label in dataset['label'].unique():
        samples = dataset[dataset['label'] == label].sample(samples_per_label, random_state=42)
        subset.append(samples)

    subset = pd.concat(subset)
    subset = subset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    subset.to_csv(path_or_buf="subdataset.csv", sep=";", index=False)


def downsample_dataset(dataset):
    print("Before downsampling: ")
    print(dataset['label'].value_counts())

    lowest_shape = float('inf')
    for label in ID2LABEL.keys():
        df_label = dataset[dataset['label'] == label]
        lowest_shape = min(lowest_shape, df_label.shape[0])

    downsampled_labels = []
    for label in ID2LABEL.keys():
        df_label = dataset[dataset['label'] == label]
        downsampled_df = df_label.sample(lowest_shape, random_state=42)
        downsampled_labels.append(downsampled_df)

    balanced_dataset = pd.concat(downsampled_labels)

    print("After downsampling: ")
    print(balanced_dataset['label'].value_counts())

    return balanced_dataset
