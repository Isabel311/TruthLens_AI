"""
TruthLens AI - Model Training Script

This script trains a Logistic Regression fake news classifier using:
- NLTK text preprocessing
- TF-IDF vectorization
- Scikit-learn LogisticRegression

Expected Kaggle dataset files:
    dataset/Fake.csv
    dataset/True.csv

Run:
    python train_model.py
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
FAKE_FILE = DATASET_DIR / "Fake.csv"
TRUE_FILE = DATASET_DIR / "True.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
VECTORIZER_FILE = BASE_DIR / "vectorizer.pkl"
METRICS_FILE = BASE_DIR / "metrics.json"


def download_nltk_resources() -> None:
    """Kept for Colab/classroom readability; preprocessing works without downloads."""
    return None


def get_stop_words() -> set[str]:
    """Use NLTK stopwords when available, otherwise fall back to scikit-learn stopwords."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)


def preprocess_text(text: str) -> str:
    """Clean text with lowercasing, punctuation removal, tokenization, and stopword removal."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = wordpunct_tokenize(text)
    stop_words = get_stop_words()
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)


def load_kaggle_dataset() -> pd.DataFrame:
    """Load Fake.csv and True.csv from the Kaggle Fake and Real News Dataset."""

    if not FAKE_FILE.exists() or not TRUE_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found. Add Kaggle files to dataset/Fake.csv and dataset/True.csv."
        )

    print("Fake file path:", FAKE_FILE)
    print("True file path:", TRUE_FILE)
    print("Fake exists:", FAKE_FILE.exists())
    print("True exists:", TRUE_FILE.exists())

    fake_df = pd.read_csv(FAKE_FILE, nrows=7000)
    true_df = pd.read_csv(TRUE_FILE, nrows=7000)

    fake_df["label"] = 0
    true_df["label"] = 1

    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    title = combined_df.get("title", pd.Series("", index=combined_df.index)).fillna("")
    body = combined_df.get("text", pd.Series("", index=combined_df.index)).fillna("")
    combined_df["content"] = title + " " + body

    return combined_df[["content", "label"]].dropna()


def load_demo_dataset() -> pd.DataFrame:
    """
    Small fallback dataset for quick smoke tests.

    This is not a replacement for the Kaggle dataset. It only lets students verify
    that the pipeline runs before adding the full CSV files.
    """
    demo_rows = [
        ("Government announces verified economic relief plan after official cabinet meeting", 1),
        ("Health ministry releases peer reviewed vaccination progress report", 1),
        ("Election commission confirms voting schedule through official press briefing", 1),
        ("NASA publishes new climate data from satellite observations", 1),
        ("Breaking celebrity says secret cure can make people immortal overnight", 0),
        ("Shocking claim world leaders replaced by clones goes viral online", 0),
        ("Miracle pill destroys all diseases in one day according to unknown blog", 0),
        ("Fake leaked report claims moon will disappear next week", 0),
    ]
    return pd.DataFrame(demo_rows, columns=["content", "label"])


def train_model(use_demo_data: bool = False) -> dict:
    """Train the classifier, save artifacts, and return evaluation metrics."""
    download_nltk_resources()

    if use_demo_data:
        data = load_demo_dataset()
        print("Using demo dataset. Add Kaggle CSV files for real training.")
    else:
        data = load_kaggle_dataset()

    data["clean_content"] = data["content"].apply(preprocess_text)
    data = data[data["clean_content"].str.strip().astype(bool)]

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_content"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "total_samples": int(len(data)),
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "classes": {"0": "Fake", "1": "Real"},
        "confusion_matrix": np.asarray(confusion_matrix(y_test, y_pred)).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Fake", "Real"],
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("Training complete.")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved vectorizer to: {VECTORIZER_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")
    return metrics


if __name__ == "__main__":
    try:
        train_model(use_demo_data=False)
    except (FileNotFoundError, MemoryError) as error:
        print(error)
        print("Training a small demo model so the Streamlit app can be tested immediately.")
        train_model(use_demo_data=True)
