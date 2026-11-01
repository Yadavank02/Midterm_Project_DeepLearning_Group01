"""
diagrams.py
Draws the architecture diagram by running a batch through the real classes in models.py and
recording each layer's output shape and parameter count. Writes artifacts/figure0_architectures.png.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import TextCNN, TextRNN, TextTransformer

BATCH, LENGTH = 64, 35


def trace(model: nn.Module, x: torch.Tensor):
    """Returns (name, type, output shape, parameters) for each top-level layer, in the order it runs."""
    rows, hooks = [], []

    def record(name, module):
        def hook(_, __, output):
            output = output[0] if isinstance(output, tuple) else output
            rows.append((name, type(module).__name__, list(output.shape)[1:],
                         sum(p.numel() for p in module.parameters())))
        return hook

    for name, module in model.named_modules():
        is_block = isinstance(module, (nn.LSTM, nn.TransformerEncoder))
        inside_block = name.startswith("transformer_encoder.")
        if name and not inside_block and (is_block or not list(module.children())):
            hooks.append(module.register_forward_hook(record(name, module)))
    model.eval()
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()
    return rows


def draw(ax, title, rows, total):
    ax.set_xlim(0, len(rows))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{title}: {total:,} parameters", loc="left", fontsize=12, fontweight="bold")
    for i, (name, kind, shape, params) in enumerate(rows):
        ax.add_patch(plt.Rectangle((i + 0.04, 0.12), 0.84, 0.72, facecolor="#EEF2F7", edgecolor="#14213D"))
        ax.text(i + 0.46, 0.66, kind, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(i + 0.46, 0.44, str(shape), ha="center", va="center", fontsize=8)
        ax.text(i + 0.46, 0.26, f"{params:,} params", ha="center", va="center", fontsize=7, color="#46536B")
        if i < len(rows) - 1:
            ax.annotate("", xy=(i + 1.04, 0.48), xytext=(i + 0.88, 0.48),
                        arrowprops={"arrowstyle": "->", "color": "#C2410C"})


def main():
    with open("artifacts/results.json") as f:
        meta = json.load(f)["SNIPS"]["meta"]
    v, c = meta["vocab_size"], meta["num_classes"]
    x = torch.randint(1, v, (BATCH, LENGTH))
    x[:, 20:] = 0
    models = [
        ("TextCNN (the three Conv1d + BatchNorm1d pairs are parallel branches, shown in run order)", TextCNN(v, num_classes=c)),
        ("TextRNN (Bi-LSTM)", TextRNN(v, num_classes=c)),
        ("TextTransformer", TextTransformer(v, num_classes=c)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.2))
    for ax, (title, model) in zip(axes, models):
        draw(ax, title, trace(model, x), sum(p.numel() for p in model.parameters()))
    fig.text(0.01, 0.005, f"Output shape excludes batch size {BATCH}. Traced from models.py, SNIPS vocabulary {v:,}, length {LENGTH}.",
             fontsize=8, color="#46536B")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("artifacts/figure0_architectures.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
