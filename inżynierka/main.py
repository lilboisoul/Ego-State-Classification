from helper import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import Trainer, TrainingArguments

_train = False
_evaluate = False
_prompt = True

if _train or _evaluate:

    df = load_dataset("data/goemotions-1000-label.csv")
    df = downsample_dataset(df)

    X = df["text"]
    y = df["label"].to_numpy()

    # Create 85/15 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X = X_train
    y = y_train
    # train_with_crossvalidation(model=bert_model, tokenizer=bert_tokenizer, X=X, y=y)
    # train_with_crossvalidation(roberta_model, roberta_tokenizer, X_train, y_train)
    if _evaluate:
        bert_model1 = BertForSequenceClassification.from_pretrained('test_model-1', num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id)
        bert_model2 = BertForSequenceClassification.from_pretrained('test_model-2', num_labels=num_labels,
                                                                    id2label=id2label,
                                                                    label2id=label2id)
        bert_model3 = BertForSequenceClassification.from_pretrained('test_model-3', num_labels=num_labels,
                                                                    id2label=id2label,
                                                                    label2id=label2id)
        evaluate_model_on_test_set(tokenizer=bert_tokenizer,
                                   evaluated_model=bert_model1,
                                   X_test=X_test,
                                   y_test=y_test)
        evaluate_model_on_test_set(tokenizer=bert_tokenizer,
                                   evaluated_model=bert_model2,
                                   X_test=X_test,
                                   y_test=y_test)
        evaluate_model_on_test_set(tokenizer=bert_tokenizer,
                                   evaluated_model=bert_model3,
                                   X_test=X_test,
                                   y_test=y_test)


    elif _train:
        train_metrics = {}
        eval_metrics = {}
        metrics = {'train_loss': [], 'eval_loss': [], 'accuracy': []}
        current_dir = f"Training-{current_date()}"
        os.makedirs(current_dir, exist_ok=True)
        with open(os.path.join(current_dir, f"training-log-{current_date()}"), 'w') as file:
            kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train_tokenized = bert_tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
                X_val_tokenized = bert_tokenizer(list(X_val), padding=True, truncation=True, max_length=512)

                train_dataset = CustomDataset(X_train_tokenized, y_train)
                eval_dataset = CustomDataset(X_val_tokenized, y_val)

                args = TrainingArguments(
                    output_dir=os.path.join(current_dir, f"fold{fold + 1}-{current_date()}"),
                    num_train_epochs=NUM_EPOCHS,
                    per_device_train_batch_size=BATCH_SIZE,
                    save_strategy="epoch",
                    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
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

            # Save or show the plot
            plot_file_path = os.path.join(current_dir, "loss-accuracy-plot.png")
            plt.savefig(plot_file_path)
        # plt.show()

if _prompt == True:
    np.set_printoptions(suppress=True)
    texts = [
        "I'm so proud of all your accomplishments.",  # NP
        "Do not do that",  # CP
        "I have realized how that works",  # A
        "I am so excited to see you!",  # FC
        "I don't want to upset you"  # AC
    ]
    trained_models = [
        BertForSequenceClassification.from_pretrained('test_model-1/',
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id=label2id),
        BertForSequenceClassification.from_pretrained('test_model-2/',
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
