"""
compare.py
Generates the report tables and comparison figures from artifacts/results.json,
artifacts/predictions.json and artifacts/costs.json. Runs without any training.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

SEEDS = ["42", "123", "999"]
CORE = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]
LABELS = {"Baseline": "Baseline", "CNN_Standard": "CNN", "RNN_BiLSTM": "Bi-LSTM", "Transformer_Standard": "Transformer"}
COLORS = {"Baseline": "#444444", "CNN_Standard": "#1b9e77", "RNN_BiLSTM": "#7570b3", "Transformer_Standard": "#d95f02"}
ABLATIONS = [
    ("CNN: remove BatchNorm", "CNN_Standard", "CNN_NoBN"),
    ("LSTM: remove backward pass", "RNN_BiLSTM", "RNN_UniLSTM"),
    ("Transformer: remove positions", "Transformer_Standard", "Transformer_NoPos"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def seed_f1(results, dataset, variant):
    runs = results[dataset]["variants"][variant]["seeds"]
    return [runs[s]["test_f1"] for s in SEEDS]


def print_tables(results, costs):
    for dataset in ["SNIPS", "ATIS"]:
        print(f"\nCORE COMPARISON: {dataset} {results[dataset]['meta']['split_sizes']}")
        print(f"{'Model':<22}{'Seed 42':>9}{'Seed 123':>10}{'Seed 999':>10}{'Mean':>9}{'Spread':>9}")
        for m in CORE:
            v = seed_f1(results, dataset, m)
            print(f"{m:<22}{v[0]:>9.4f}{v[1]:>10.4f}{v[2]:>10.4f}{np.mean(v):>9.4f}{np.ptp(v):>9.4f}")

    print("\nTUNED SETTINGS (chosen on validation macro F1)")
    for dataset in ["SNIPS", "ATIS"]:
        for m in CORE:
            config = results[dataset]["variants"][m]["config"]
            print(f"{dataset:<7}{m:<22}lr={config['lr']:<8}weight_decay={config['weight_decay']}")

    print("\nABLATIONS: SNIPS (same split, per-seed change = ablated minus standard)")
    print(f"{'Change':<32}{'Standard':>10}{'Ablated':>10}{'Change':>9}   Per seed")
    for name, std, abl in ABLATIONS:
        s, a = seed_f1(results, "SNIPS", std), seed_f1(results, "SNIPS", abl)
        per_seed = ", ".join(f"{y - x:+.4f}" for x, y in zip(s, a))
        print(f"{name:<32}{np.mean(s):>10.4f}{np.mean(a):>10.4f}{np.mean(a) - np.mean(s):>+9.4f}   {per_seed}")

    print(f"\nINFERENCE COST: {costs['hardware']}")
    print(f"{'Model':<22}{'Parameters':>12}{'ms/sample':>11}{'Peak RSS MB':>13}{'Increase MB':>13}")
    for m in CORE:
        runs = costs["models"][m].values()
        print(f"{m:<22}{next(iter(runs))['params']:>12,}"
              f"{np.mean([r['latency_ms'] for r in runs]):>11.4f}"
              f"{np.mean([r['peak_rss_mb'] for r in runs]):>13.1f}"
              f"{np.mean([r['peak_rss_increase_mb'] for r in runs]):>13.1f}")


def figure_loss(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dataset in zip(axes, ["SNIPS", "ATIS"]):
        for m in CORE:
            curve = results[dataset]["variants"][m]["loss_curves"]["42"]
            ax.plot(range(1, len(curve) + 1), curve, label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.set_title(f"{dataset} training loss (seed 42)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("artifacts/figure1_loss_convergence.png", dpi=200)
    plt.close()


def figure_seeds(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dataset in zip(axes, ["SNIPS", "ATIS"]):
        for i, m in enumerate(CORE):
            v = seed_f1(results, dataset, m)
            ax.scatter([i] * 3, v, color=COLORS[m], s=55, alpha=0.8, zorder=3)
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=COLORS[m], linewidth=2.5)
        ax.set_xticks(range(len(CORE)))
        ax.set_xticklabels([LABELS[m] for m in CORE])
        ax.set_title(f"{dataset} test macro F1: dots are seeds, bars are means")
        ax.set_ylabel("Test macro F1")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/figure2_dataset_generalization.png", dpi=200)
    plt.close()


def figure_cost(results, costs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in CORE:
        runs = costs["models"][m].values()
        lat = [r["latency_ms"] for r in runs]
        f1 = seed_f1(results, "SNIPS", m)
        ax.errorbar(np.mean(lat), np.mean(f1), xerr=np.ptp(lat) / 2, yerr=np.ptp(f1) / 2,
                    fmt="o", color=COLORS[m], capsize=4, label=LABELS[m])
    ax.set_xlabel("Inference time per sample (ms)")
    ax.set_ylabel("SNIPS test macro F1")
    ax.set_title("Cost against score: bars show the range over three seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("artifacts/figure3_cost_vs_f1.png", dpi=200)
    plt.close()


def figure_confusions(results, predictions):
    classes = results["SNIPS"]["meta"]["classes"]
    labels = predictions["SNIPS"]["labels"]
    names = {"Baseline": "baseline", "CNN_Standard": "cnn", "RNN_BiLSTM": "rnn", "Transformer_Standard": "transformer"}
    for m in CORE:
        cm = confusion_matrix(labels, predictions["SNIPS"][m]["42"], labels=range(len(classes)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"{LABELS[m]} confusion matrix, SNIPS test (trained model, seed 42)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground truth")
        plt.tight_layout()
        plt.savefig(f"artifacts/{names[m]}_confusion_matrix.png", dpi=200)
        plt.close()


def main():
    os.makedirs("artifacts", exist_ok=True)
    results = load("artifacts/results.json")
    predictions = load("artifacts/predictions.json")
    costs = load("artifacts/costs.json")
    print_tables(results, costs)
    figure_loss(results)
    figure_seeds(results)
    figure_cost(results, costs)
    figure_confusions(results, predictions)
    print("\nFigures written to artifacts/")


if __name__ == "__main__":
    main()
