
import sys
import csv
import ctype
try:
    safe_limit = min(sys.maxsize, ctypes.c_long.max)
except Exception:
    
    safe_limit = (2**31) - 1

try:
    csv.field_size_limit(safe_limit)
except OverflowError:
    csv.field_size_limit((2**31) - 1)

import os
import glob
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers

import sys, csv

csv.field_size_limit(sys.maxsize)
def load_all_csvs(data_dir: str) -> pd.DataFrame:

    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8", engine="python")
        df["__source_file"] = os.path.basename(p)
        dfs.append(df)

    data = pd.concat(dfs, axis=0, ignore_index=True)
    return data


def basic_text_preprocess(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    
    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    return df


def build_vectorizer(texts: List[str], vocab_size: int = 20000, seq_len: int = 200) -> keras.layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace"
    )
    text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(128)
    vectorizer.adapt(text_ds)
    return vectorizer


def build_model(vocab_size: int = 20000, seq_len: int = 200, embedding_dim: int = 64, num_classes: int = 2) -> keras.Model:
    inputs = keras.Input(shape=(seq_len,), dtype=tf.int32, name="token_ids")
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embed")(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_save(
    data_dir: str,
    text_col: str,
    label_col: str,
    save_path: str,
    vocab_size: int = 20000,
    seq_len: int = 200,
    batch_size: int = 64,
    epochs: int = 5,
    val_split: float = 0.1
):
    # 1) Load all datasets
    df = load_all_csvs(data_dir)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and/or {label_col} not found in data.")

    # 2) Basic preprocessing
    df = basic_text_preprocess(df, text_col)

    # Convert labels: if they are 'ham'/'spam' -> map to integers
    if df[label_col].dtype == object:
        label_map = {lbl: i for i, lbl in enumerate(sorted(df[label_col].unique()))}
        df[label_col] = df[label_col].map(label_map)
        print(f"Label map: {label_map}")
        num_classes = len(label_map)
    else:
        num_classes = int(df[label_col].nunique())
        print(f"Detected {num_classes} classes")

    texts = df[text_col].tolist()
    labels = df[label_col].astype(int).values

    # 3) Vectorizer
    vectorizer = build_vectorizer(texts, vocab_size=vocab_size, seq_len=seq_len)

    # Vectorize
    text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(128)
    tokenized = []
    for batch in text_ds:
        tokenized.append(vectorizer(batch).numpy())
    X = np.vstack(tokenized)

    # 4) Train/Val split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]

    # 5) Build model
    model = build_model(vocab_size=vocab_size, seq_len=seq_len, embedding_dim=64, num_classes=num_classes)

    # 6) Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # 7) Save model with Keras model.save(...)
    # This saves the full Keras model (architecture + weights + optimizer state if any).
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # 8) Also save the vectorizer as a Keras layer (SavedModel supports it)
    vec_save_path = os.path.splitext(save_path)[0] + "_vectorizer"
    tf.saved_model.save(vectorizer, vec_save_path)
    print(f"Vectorizer saved to: {vec_save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text classifier on multiple CSV datasets and save with model.save")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CSV datasets")
    parser.add_argument("--text_col", type=str, default="v2", help="Text column name")
    parser.add_argument("--label_col", type=str, default="v1", help="Label column name")
    parser.add_argument("--save_path", type=str, default="model/saved_model.h5", help="Where to save the model (e.g., model/saved_model.h5)")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_dir=args.data_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        save_path=args.save_path,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split
    )