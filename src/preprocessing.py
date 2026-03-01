import os
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BERT tokenizer
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

MAX_LEN = 128


def load_dataset(filename):
    data_path = os.path.join(BASE_DIR, "data", filename)
    return pd.read_csv(data_path)


def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )


def prepare_dataloaders(csv_filename, test_size=0.2):
    df = load_dataset(csv_filename)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["anxiety_level"],
        test_size=test_size,
        stratify=df["anxiety_level"],
        random_state=42
    )

    train_encodings = tokenize_texts(X_train)
    val_encodings = tokenize_texts(X_val)

    train_labels = torch.tensor(y_train.values)
    val_labels = torch.tensor(y_val.values)

    return train_encodings, val_encodings, train_labels, val_labels
