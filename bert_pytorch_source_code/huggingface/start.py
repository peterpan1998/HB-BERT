import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, Trainer, TrainingArguments

from config import TRAINING_CONFIG, FILE_PATHS, DATA_CONFIG, DEVICE_CONFIG, EVAL_CONFIG
from main_mnli import CustomBertModel


def load_parquet_file(parquet_file):
    return pd.read_parquet(parquet_file)


def build_datasets():
    train_df = load_parquet_file(FILE_PATHS['train_file'])
    val_df = load_parquet_file(FILE_PATHS['validation_file'])
    test_df = load_parquet_file(FILE_PATHS['test_file'])
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)
    ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})
    print(ds['train'].column_names)
    print(ds['validation'].column_names)
    return ds


def tokenize_datasets(dataset):
    tokenizer = BertTokenizer.from_pretrained(FILE_PATHS['local_model_path'])

    def preprocess_function(examples):
        return tokenizer(
            examples[DATA_CONFIG['text_columns'][0]],
            examples[DATA_CONFIG['text_columns'][1]],
            padding=DATA_CONFIG['padding'],
            truncation=DATA_CONFIG['truncation'],
            max_length=DATA_CONFIG['max_length'],
        )

    print("Preprocessing the dataset...")
    encoded = dataset.map(preprocess_function, batched=True)
    print("Dataset preprocessing completed.")
    return encoded


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = torch.argmax(torch.tensor(predictions), dim=-1)
    acc = accuracy_score(labels, preds)
    return {"eval_accuracy": acc}


def main():
    dataset = build_datasets()
    encoded_dataset = tokenize_datasets(dataset)
    train_dataset = encoded_dataset['train']
    eval_dataset = encoded_dataset['validation']

    print("Loading BERT model for sequence classification...")
    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else "cpu")
    model = CustomBertModel.from_pretrained(FILE_PATHS['local_model_path'], num_labels=TRAINING_CONFIG['num_labels'])
    model.to(device)
    print("Model loaded successfully.")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG['output_dir'],
        num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['per_device_eval_batch_size'],
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        logging_dir=TRAINING_CONFIG['logging_dir'],
        logging_steps=TRAINING_CONFIG['logging_steps'],
        metric_for_best_model=TRAINING_CONFIG['metric_for_best_model'],
        eval_strategy=EVAL_CONFIG['eval_strategy'],
        save_strategy=EVAL_CONFIG['save_strategy'],
        load_best_model_at_end=EVAL_CONFIG['load_best_model_at_end'],
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()
