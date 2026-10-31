"""
main.py
Midterm Project.
Tunes every family on the same validation grid, then runs Baseline, CNN, RNN and Transformer
+ Ablations over Seeds 42, 123, 999 on one fixed split per dataset.
Exports results to artifacts/results.json and test predictions to artifacts/predictions.json.
Both are saved after every run, so an interrupted run resumes where it stopped.
"""

import copy
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from data_pipeline import prepare_dataset_pipeline
from models import BagOfWordsLogReg, TextCNN, TextRNN, TextTransformer

SEEDS = [42, 123, 999]
SPLIT_SEED = 42
EPOCHS = 20
TUNING_SEED = 42
BATCH_SIZE = 64
GRID = [
    {"lr": 0.001, "weight_decay": 0.0},
    {"lr": 0.001, "weight_decay": 0.01},
    {"lr": 0.0003, "weight_decay": 0.0},
    {"lr": 0.0003, "weight_decay": 0.01},
]
RESULTS_PATH = "artifacts/results.json"
PREDICTIONS_PATH = "artifacts/predictions.json"

DEVICE = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))

# variant -> (family it takes tuned settings from, model class, constructor arguments)
VARIANTS = {
    "Baseline": ("Baseline", BagOfWordsLogReg, {}),
    "CNN_Standard": ("CNN_Standard", TextCNN, {"use_batch_norm": True}),
    "CNN_NoBN": ("CNN_Standard", TextCNN, {"use_batch_norm": False}),
    "RNN_BiLSTM": ("RNN_BiLSTM", TextRNN, {"bidirectional": True}),
    "RNN_UniLSTM": ("RNN_BiLSTM", TextRNN, {"bidirectional": False}),
    "Transformer_Standard": ("Transformer_Standard", TextTransformer, {"use_positional": True}),
    "Transformer_NoPos": ("Transformer_Standard", TextTransformer, {"use_positional": False}),
}
FAMILIES = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]
DATASET_VARIANTS = {"SNIPS": list(VARIANTS), "ATIS": FAMILIES}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model: nn.Module, loader) -> Tuple[List[int], List[int]]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(DEVICE))
            preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            targets.extend(b["label"].tolist())
    return preds, targets


def macro_f1(model: nn.Module, loader) -> float:
    preds, targets = predict(model, loader)
    return f1_score(targets, preds, average="macro", zero_division=0)


def train_model(model: nn.Module, train_loader, val_loader, lr: float, weight_decay: float) -> Tuple[Dict, List[float]]:
    """Trains on the train split and restores the best validation epoch. Never reads the test split."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1, best_epoch, best_weights = -1.0, 0, None
    loss_history = []
    started = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for b in train_loader:
            x, y = b["input_ids"].to(DEVICE), b["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        scheduler.step()
        loss_history.append(epoch_loss / len(train_loader.dataset))
        v_f1 = macro_f1(model, val_loader)
        if v_f1 > best_val_f1:
            best_val_f1, best_epoch = v_f1, epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return {
        "val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "seconds_per_epoch": (time.perf_counter() - started) / EPOCHS,
    }, loss_history


def build(variant: str, vocab_size: int, num_classes: int) -> nn.Module:
    _, model_class, params = VARIANTS[variant]
    return model_class(vocab_size, num_classes=num_classes, **params).to(DEVICE)


def load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def run_dataset(dataset: str, results: Dict, predictions: Dict):
    loaders, vocab, meta = prepare_dataset_pipeline(
        dataset.lower(), seed=SPLIT_SEED, batch_size=BATCH_SIZE, run_eda=True
    )
    train_loader, val_loader, test_loader = loaders
    v_size, n_cls = len(vocab), meta["num_classes"]
    out = results.setdefault(dataset, {"tuning": {}, "variants": {}})
    out["meta"] = meta
    preds_out = predictions.setdefault(dataset, {})

    # Step 1: the same grid, seed and budget for every family, decided on validation only
    for family in FAMILIES:
        rows = out["tuning"].setdefault(family, [])
        for i, config in enumerate(GRID):
            if i < len(rows):
                continue
            set_seed(TUNING_SEED)
            res, _ = train_model(build(family, v_size, n_cls), train_loader, val_loader, **config)
            rows.append({**config, **res})
            save_json(RESULTS_PATH, results)
            print(f"{dataset} tune {family} {config} val_f1={res['val_f1']:.4f}", flush=True)

    # Step 2: final runs, with ablations holding their family's tuned settings fixed
    os.makedirs("checkpoints", exist_ok=True)
    for variant in DATASET_VARIANTS[dataset]:
        family = VARIANTS[variant][0]
        best = max(out["tuning"][family], key=lambda row: row["val_f1"])
        config = {"lr": best["lr"], "weight_decay": best["weight_decay"]}
        entry = out["variants"].setdefault(variant, {"config": config, "seeds": {}, "loss_curves": {}})
        for seed in SEEDS:
            if str(seed) in entry["seeds"]:
                continue
            set_seed(seed)
            model = build(variant, v_size, n_cls)
            res, curve = train_model(model, train_loader, val_loader, **config)
            preds, targets = predict(model, test_loader)
            res["test_acc"] = accuracy_score(targets, preds)
            res["test_f1"] = f1_score(targets, preds, average="macro", zero_division=0)
            res["params"] = count_parameters(model)
            entry["seeds"][str(seed)] = res
            entry["loss_curves"][str(seed)] = curve
            preds_out.setdefault(variant, {})[str(seed)] = preds
            preds_out["labels"] = targets
            torch.save(model.state_dict(), f"checkpoints/{dataset}_{variant}_{seed}.pt")
            save_json(RESULTS_PATH, results)
            save_json(PREDICTIONS_PATH, predictions)
            print(f"{dataset} {variant} seed={seed} test_f1={res['test_f1']:.4f}", flush=True)


def run_full_suite():
    os.makedirs("artifacts", exist_ok=True)
    results = load_json(RESULTS_PATH)
    if "variants" not in results.get("SNIPS", {"variants": 0}):
        results = {}  # results from the earlier pipeline can't be resumed
    predictions = load_json(PREDICTIONS_PATH)
    for dataset in DATASET_VARIANTS:
        run_dataset(dataset, results, predictions)
    print("\nBenchmark complete. Outputs saved to artifacts/")


if __name__ == "__main__":
    run_full_suite()
