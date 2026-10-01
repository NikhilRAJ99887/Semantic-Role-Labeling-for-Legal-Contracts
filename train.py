import ast
import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

from config import *


def load_data():
    df = pd.read_csv(DATA_FILE)
    # CSV stores Python-list strings such as "['The', 'developer']".
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(ast.literal_eval)
    return DatasetDict({
        split: Dataset.from_pandas(df[df["split"] == split].reset_index(drop=True), preserve_index=False)
        for split in ["train", "validation", "test"]
    })


def get_labels(ds):
    labels = set()
    for tags in ds["train"]["tags"]:
        labels.update(tags)
    for tags in ds["validation"]["tags"]:
        labels.update(tags)
    for tags in ds["test"]["tags"]:
        labels.update(tags)
    labels = sorted(labels, key=lambda x: (x != "O", x))
    return labels


def main():
    set_seed(SEED)
    datasets = load_data()
    labels = get_labels(datasets)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("Labels:", labels)
    print("Dataset sizes:", {k: len(v) for k, v in datasets.items()})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        all_labels = []
        for i, word_labels in enumerate(examples["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[word_labels[word_id]])
                else:
                    # Give continuation subwords the same entity label.
                    # This keeps training simple for this academic project.
                    label_ids.append(label2id[word_labels[word_id]])
                previous_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets["train"].column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels_batch = eval_pred
        predictions = np.argmax(logits, axis=2)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(predictions, labels_batch):
            p_seq, l_seq = [], []
            for p, l in zip(pred, lab):
                if l != -100:
                    p_seq.append(id2label[int(p)])
                    l_seq.append(id2label[int(l)])
            true_predictions.append(p_seq)
            true_labels.append(l_seq)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nValidation metrics:")
    print(trainer.evaluate(tokenized["validation"]))

    print("\nTest metrics:")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    # Detailed test classification report
    output = trainer.predict(tokenized["test"])
    preds = np.argmax(output.predictions, axis=2)
    true_predictions, true_labels = [], []
    for pred, lab in zip(preds, output.label_ids):
        p_seq, l_seq = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                p_seq.append(id2label[int(p)])
                l_seq.append(id2label[int(l)])
        true_predictions.append(p_seq)
        true_labels.append(l_seq)
    print("\nDetailed test report:\n")
    print(classification_report(true_labels, true_predictions, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    import json
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2)
    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
