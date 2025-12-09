import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# CONFIG
DATA_CSV = os.path.join("data", "sms_spam.csv")  # expects columns: v1 (label), v2 (text)
OUTPUT_TFLITE = os.path.join("model", "model.tflite")
VOCAB_SIZE = 10000
SEQ_LEN = 50
EPOCHS = 6
BATCH_SIZE = 64

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http[s]?://\S+", " URL ", s)
    s = re.sub(r"[^a-z0-9\s@]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset():
    df = pd.read_csv(DATA_CSV, encoding="latin-1")
    # Some versions of the dataset have extra columns; keep only label/text
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["y"] = (df["label"].str.lower().str.strip() == "spam").astype(int)
    return df[["text", "y"]]

def build_model():
    inputs = tf.keras.Input(shape=(None,), dtype=tf.string, name="text")
    # TextVectorization to embed in-model for easy deployment
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        standardize=None,  # we already cleaned
    )
    # We will adapt on training corpus inside the script
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, 64)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, vectorizer

def main():
    os.makedirs("model", exist_ok=True)
    df = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(df["text"].values, df["y"].values, test_size=0.2, random_state=42, stratify=df["y"].values)

    # Build model with in-graph vectorizer
    model, vectorizer = build_model()
    # Fit the vectorizer on train data
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {acc:.4f}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimize for size/speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {OUTPUT_TFLITE}")

if __name__ == "__main__":
    main()