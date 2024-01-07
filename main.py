import helper.helper
from helper.dataset import *
from sklearn.model_selection import train_test_split, KFold
from transformers import Trainer, TrainingArguments
import time
from datetime import timedelta

start_time = time.monotonic()
_train = True
_evaluate = False
_prompt = False

# DATASET REMAPPING
# original_dataset_to_mapped_dataset("data/dataset-to-remap.csv", "data/dataset-functional.csv")
# original_dataset_to_mapped_dataset("data/dataset-to-remap.csv", "data/dataset-structural.csv")

# SUBSET GENERATION
# df = pd.read_csv("data/dataset-structural.csv", delimiter=";")
# subset_generator(df, 1000)

# HYPERPARAMETER SEARCH
# hyperparameter_train_loop()

if _train or _evaluate:
    dataset_path = "data/dataset-structural.csv"
    dataset = pd.read_csv(dataset_path, delimiter=";")
    dataset = downsample_dataset(dataset)

    dataset_text = dataset["text"]
    dataset_label = dataset["label"].to_numpy()

    text_training, text_test, label_training, label_test = train_test_split(dataset_text,
                                                                            dataset_label,
                                                                            test_size=0.2,
                                                                            stratify=dataset_label,
                                                                            random_state=42)
    text = text_training
    label = label_training

    if _train:
        train_metrics = {}
        eval_metrics = {}
        metrics = {'train_loss': [], 'eval_loss': [], 'accuracy': []}
        current_dir = f"Training-{helper.helper.current_date()}"
        helper.helper.os.makedirs(current_dir, exist_ok=True)

        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(text_training, label_training)):
            text_train, text_validation = text_training.iloc[train_idx], text_training.iloc[val_idx]
            label_train, label_validation = label_training[train_idx], label_training[val_idx]

            text_train_tokenized =      bert_tokenizer(list(text_train),
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=512)
            text_validation_tokenized = bert_tokenizer(list(text_validation),
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=512)

            train_dataset =      CustomDataset(text_train_tokenized, label_train)
            validation_dataset = CustomDataset(text_validation_tokenized, label_validation)

            args = TrainingArguments(
                output_dir=helper.helper.os.path.join(current_dir, f"fold{fold + 1}-{helper.helper.current_date()}"),
                learning_rate=LEARNING_RATE,
                warmup_ratio=WARMUP_RATIO,
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                save_strategy="epoch",
            )
            trainer = Trainer(
                model=bert_model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=helper.helper.compute_metrics_training
            )
            print(f"Training fold {fold + 1}...")

            train_result = trainer.train()
            eval_result = trainer.evaluate()

            metrics['train_loss'].append(train_result.metrics['train_loss'])
            metrics['eval_loss'].append(eval_result['eval_loss'])
            metrics['accuracy'].append(eval_result['eval_accuracy'])

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
        plt.plot(range(1, NUM_FOLDS + 1), metrics['accuracy'], marker='o', color='r')
        plt.title("Accuracy per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")

        plt.tight_layout()

        plot_file_path = helper.helper.os.path.join(current_dir, "loss-accuracy-plot.png")
        plt.savefig(plot_file_path)

    elif _evaluate:
        text_test_tokenized = bert_tokenizer(list(text_test),
                                             padding=True,
                                             truncation=True,
                                             max_length=512)
        evaluation_model_path = r'structural-model'
        evaluation_model = BertForSequenceClassification.from_pretrained(evaluation_model_path,
                                                                   num_labels=NUM_LABELS,
                                                                   id2label=ID2LABEL,
                                                                   label2id=LABEL2ID).to('cuda')

        test_dataset = CustomDataset(text_test_tokenized, label_test)

        trainer = Trainer(
            model=evaluation_model,
            eval_dataset=test_dataset,
            compute_metrics=helper.helper.compute_metrics_evaluation
        )
        print(trainer.evaluate())

if _prompt:
    texts = [
        "I'm so proud of all your accomplishments.",  # NP
        "Do not do that",  # CP
        "I am so excited to see you!",  # FC
        "I don't want to upset you",  # AC
    ]
    trained_model_path = 'functional-model'
    trained_model = BertForSequenceClassification.from_pretrained(
            trained_model_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID).to('cuda')

    predictions_dir = 'predictions-functional'
    helper.helper.os.makedirs(predictions_dir, exist_ok=True)

    for i, text in enumerate(texts):
        predictions = helper.helper.get_predictions(text=text,
                                                    tokenizer=bert_tokenizer,
                                                    model=trained_model)[0]

        prediction_index = helper.helper.np.argmax(predictions)
        prediction_value = predictions[prediction_index]

        result = ID2LABEL[prediction_index]

        helper.helper.np.set_printoptions(suppress=True)
        print(f"Text: {text} | label: {result}")

        plt.figure(figsize=(12, 6))
        plt.bar(LABELS, predictions)
        plt.title(f"Text input: '{text}'")
        plt.ylabel('Probability')
        plt.ylim([0, 1])

        plot_filename = helper.helper.os.path.join(predictions_dir, f"prediction_text_{i + 1}.png")

        plt.savefig(plot_filename)
        plt.close()

    print("All plots saved.")

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
