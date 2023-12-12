from config.preprocessing import original_dataset_to_mapped_dataset
from helper import *
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from transformers import Trainer, TrainingArguments
import time
from datetime import timedelta
start_time = time.monotonic()
_train = True
_evaluate = False
_prompt = False
#DATASET REMAPPING
#original_dataset_to_mapped_dataset("data/dataset-to-remap.csv", "data/dataset-functional.csv")
#original_dataset_to_mapped_dataset("data/dataset-to-remap.csv", "data/dataset-structural.csv")
#HYPERPARAMETER SEARCH
#hyperparameter_train_loop()





if _train or _evaluate:
    df = pd.read_csv("data/dataset-structural.csv", delimiter=";")
    #df = load_dataset("data/goemotions-1000-label.csv")
    df = downsample_dataset(df)

    X = df["text"]
    y = df["label"].to_numpy()

    # Create 85/15 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X = X_train
    y = y_train

    if _train:
        train_metrics = {}
        eval_metrics = {}

        metrics = {'train_loss': [], 'eval_loss': [], 'accuracy': [], 'f1': []}

        current_dir = f"Training-{current_date()}"
        os.makedirs(current_dir, exist_ok=True)

        with open(os.path.join(current_dir, f"training-log-{current_date()}"), 'w') as file:
            kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train_tokenized = bert_tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
                X_val_tokenized = bert_tokenizer(list(X_val), padding=True, truncation=True, max_length=512)

                train_dataset = CustomDataset(X_train_tokenized, y_train)
                eval_dataset = CustomDataset(X_val_tokenized, y_val)

                args = TrainingArguments(
                    output_dir=os.path.join(current_dir, f"fold{fold + 1}-{current_date()}"),
                    learning_rate=1e-5,
                    warmup_ratio=0.1,
                    num_train_epochs=NUM_EPOCHS,
                    per_device_train_batch_size=BATCH_SIZE,
                    save_strategy="epoch",
                    evaluation_strategy="epoch",
                )

                trainer = Trainer(
                    model=bert_model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics
                )
                print(f"Training fold {fold + 1}...")
                file.write(f"Training fold {fold + 1}\n")

                train_result = trainer.train()
                file.write(f"Train {fold + 1}\n")
                for key, val in train_result.metrics.items():
                    file.write(f"{key} : {val}\n")

                eval_result = trainer.evaluate()
                file.write(f"Eval {fold + 1}\n")
                for key, val in eval_result.items():
                    file.write(f"{key} : {val}\n")

                # save_evaluation_results(eval_result, file)
                metrics['train_loss'].append(train_result.metrics['train_loss'])
                metrics['eval_loss'].append(eval_result['eval_loss'])
                metrics['accuracy'].append(eval_result['eval_accuracy'])
                metrics['f1'].append(eval_result['eval_f1'])

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

            plot_file_path = os.path.join(current_dir, "loss-accuracy-plot.png")
            plt.savefig(plot_file_path)

            # X_test_tokenized = bert_tokenizer(list(X_test), padding=True, truncation=True, max_length=512)
            #
            # test_dataset = CustomDataset(X_test_tokenized, y_test)
            # trainer = Trainer(
            #     model=trainer.state.best_model_checkpoint,
            #     args=args,
            #     eval_dataset=test_dataset,
            #     compute_metrics=compute_metrics
            # )
            # evaluation = trainer.evaluate()
    elif _evaluate:
        X_test_tokenized = bert_tokenizer(list(X_test), padding=True, truncation=True, max_length=512)
        eval_model = r'functional-model'
        eval_model=BertForSequenceClassification.from_pretrained(eval_model, num_labels=num_labels, id2label=id2label,
                                                           label2id=label2id).to('cuda')

        test_dataset = CustomDataset(X_test_tokenized, y_test)
        trainer = Trainer(
            model=eval_model,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        print(trainer.evaluate())
        # plt.show()

if _prompt == True:
    np.set_printoptions(suppress=True)
    texts = [
        "I'm so proud of all your accomplishments.",  # NP
        "Do not do that",  # CP
        "I have realized how that works",  # A
        "I am so excited to see you!",  # FC
        "I don't want to upset you",  # AC
        "In my opinion it is possibly a right solution",
        "I am surprised that worked out the way it did, wow"
    ]
    trained_models = [
        BertForSequenceClassification.from_pretrained('test_model-1/',
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id=label2id),
        BertForSequenceClassification.from_pretrained('test_model-2/',
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id=label2id),
        BertForSequenceClassification.from_pretrained('test_model-3/',
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id=label2id),
        BertForSequenceClassification.from_pretrained(r'Training-08-12-2023-12-06\fold5-08-12-2023-16-53\checkpoint-14676',
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id=label2id)
    ]
    predictions_dir = 'predictions'
    os.makedirs(predictions_dir, exist_ok=True)

    for i, _model in enumerate(trained_models):
        current_model = _model.to('cuda')

        for j, text in enumerate(texts):
            predictions = get_predictions(text=text, tokenizer=bert_tokenizer, model=current_model)[0]

            plt.figure(figsize=(12, 6))
            plt.bar(config.globals.labels, predictions)
            plt.title(f"Text input: '{text}'")
            plt.ylabel('Probability')
            plt.ylim([0, 1])

            # Generate a filename for each plot
            plot_filename = os.path.join(predictions_dir, f"model{i + 1}_text_{j + 1}.png")

            # Save the plot to the file
            plt.savefig(plot_filename)
            plt.close()  # Close the plot to free memory

    print("All plots saved.")
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))