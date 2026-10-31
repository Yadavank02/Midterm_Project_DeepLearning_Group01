"""
data_pipeline.py
Data extraction, exploratory data analysis (EDA), vocabulary construction,
and PyTorch DataLoaders for SNIPS (Primary) and ATIS (Secondary) datasets.
"""

import os
import re
from collections import Counter
from typing import Dict, List, Tuple

# --- ADD THESE TWO LINES AT THE TOP ---
import matplotlib
matplotlib.use("Agg")  # Prevents macOS interactive GUI event loop hang
import matplotlib.pyplot as plt
# -------------------------------------
import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def clean_and_tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    return re.findall(r"\w+|[^\w\s]", text)


class Vocabulary:
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.pad_idx = self.add_token(self.pad_token)
        self.unk_idx = self.add_token(self.unk_token)

    def add_token(self, token: str) -> int:
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            return idx
        return self.word2idx[token]

    def build_vocab(self, texts: List[str], max_vocab_size: int = 8000, min_freq: int = 2):
        counter = Counter()
        for text in texts:
            counter.update(clean_and_tokenize(text))
        for token, count in counter.most_common(max_vocab_size):
            if count >= min_freq:
                self.add_token(token)

    def numericalize(self, text: str) -> List[int]:
        tokens = clean_and_tokenize(text)
        return [self.word2idx.get(tok, self.unk_idx) for tok in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 35):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids = self.vocab.numericalize(self.texts[idx])
        seq_len = min(len(token_ids), self.max_len)
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [self.vocab.pad_idx] * (self.max_len - len(token_ids))

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(seq_len, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_raw_dataset(dataset_name: str = "snips") -> Tuple[List[str], List[str]]:
    """
    Pulls either SNIPS or ATIS from Hugging Face hub.
    Dynamically identifies text and label column names to avoid key errors.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "snips":
        ds = load_dataset("benayas/snips")
    elif dataset_name == "atis":
        ds = load_dataset("tuetschek/atis")
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'snips' or 'atis'.")

    # Combine available splits (e.g., train and test)
    splits = [ds[s] for s in ["train", "test", "validation"] if s in ds]
    
    # 1. Identify text column name
    sample_cols = splits[0].column_names
    text_col = next((c for c in ["text", "utterance", "sentence"] if c in sample_cols), None)
    if not text_col:
        raise KeyError(f"Could not find text column in available columns: {sample_cols}")

    # 2. Identify label column name ('category' in benayas/snips, 'intent' in atis)
    label_col = next((c for c in ["category", "intent", "label", "label_text"] if c in sample_cols), None)
    if not label_col:
        raise KeyError(f"Could not find label column in available columns: {sample_cols}")

    texts = []
    labels = []
    for split in splits:
        texts.extend([str(t) for t in split[text_col]])
        labels.extend([str(l) for l in split[label_col]])

    return texts, labels


def run_data_analysis(texts: List[str], labels: List[str], dataset_name: str, output_dir: str = "artifacts"):
    """Generates Class Distribution and Token Length Distribution plots for Section 1 of Report."""
    os.makedirs(output_dir, exist_ok=True)
    lengths = [len(clean_and_tokenize(t)) for t in texts]
    label_counts = Counter(labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(list(label_counts.keys()), list(label_counts.values()), color="#2b5c8f")
    axes[0].set_title(f"{dataset_name.upper()} Class Distribution (Total: {len(texts)})")
    axes[0].set_xlabel("Number of Samples")

    axes[1].hist(lengths, bins=25, color="#d95f02", edgecolor="black")
    axes[1].set_title(f"{dataset_name.upper()} Sequence Length Distribution")
    axes[1].set_xlabel("Tokens per Sample")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_eda_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def prepare_dataset_pipeline(
    dataset_name: str = "snips",
    seed: int = 42,
    batch_size: int = 64,
    max_len: int = 35,
    max_vocab_size: int = 8000,
    run_eda: bool = False
):
    texts, raw_labels = load_raw_dataset(dataset_name)

    # Drop repeated utterances first, so no text can land in more than one split
    seen = set()
    unique_rows = []
    for text, label in zip(texts, raw_labels):
        key = " ".join(clean_and_tokenize(text))
        if key not in seen:
            seen.add(key)
            unique_rows.append((text, label))
    texts = [text for text, _ in unique_rows]
    raw_labels = [label for _, label in unique_rows]

    # Filter classes with fewer than 10 unique samples so each split gets at least one of every class
    label_counts = Counter(raw_labels)
    valid_indices = [i for i, lbl in enumerate(raw_labels) if label_counts[lbl] >= 10]
    texts = [texts[i] for i in valid_indices]
    raw_labels = [raw_labels[i] for i in valid_indices]

    if run_eda:
        run_data_analysis(texts, raw_labels, dataset_name)

    unique_labels = sorted(list(set(raw_labels)))
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = [label2idx[lbl] for lbl in raw_labels]

    # Stratified 70/15/15 train/val/test split
    train_texts, temp_texts, train_y, temp_y = train_test_split(
        texts, numeric_labels, test_size=0.30, random_state=seed, stratify=numeric_labels
    )
    val_texts, test_texts, val_y, test_y = train_test_split(
        temp_texts, temp_y, test_size=0.50, random_state=seed, stratify=temp_y
    )

    # Vocabulary fit strictly on training set
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, max_vocab_size=max_vocab_size, min_freq=2)

    train_ds = TextClassificationDataset(train_texts, train_y, vocab, max_len=max_len)
    val_ds = TextClassificationDataset(val_texts, val_y, vocab, max_len=max_len)
    test_ds = TextClassificationDataset(test_texts, test_y, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # TF-IDF vectors for the classical Baseline
    # 7. Generate unigram Bag-of-Words features for defensible Baseline model
    tfidf = TfidfVectorizer(max_features=max_vocab_size, ngram_range=(1, 1), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
    X_val_tfidf = tfidf.transform(val_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()

    metadata = {
        "dataset_name": dataset_name,
        "num_classes": len(unique_labels),
        "vocab_size": len(vocab),
        "classes": unique_labels,
        "split_sizes": {"train": len(train_texts), "val": len(val_texts), "test": len(test_texts)}
    }

    baseline_data = {
        "X_train": X_train_tfidf, "y_train": np.array(train_y),
        "X_val": X_val_tfidf,     "y_val": np.array(val_y),
        "X_test": X_test_tfidf,   "y_test": np.array(test_y)
    }

    return (train_loader, val_loader, test_loader), vocab, metadata, baseline_data