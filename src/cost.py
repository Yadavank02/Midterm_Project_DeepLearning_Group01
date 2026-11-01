"""
cost.py
Measures parameters, inference time per sample and peak memory for every core model, loading the
trained checkpoints. Each measurement runs in a fresh process, so peak memory belongs to one model.
Writes artifacts/costs.json.
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

import torch

BATCH = 64
LENGTH = 35
REPS = 200
SEEDS = [42, 123, 999]
FAMILIES = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def measure_one(variant: str, seed: int) -> dict:
    from main import VARIANTS, count_parameters

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    state = torch.load(f"checkpoints/SNIPS_{variant}_{seed}.pt", map_location="cpu")
    first_key = "embedding.weight" if "embedding.weight" in state else "fc.weight"
    vocab_size = state[first_key].shape[0 if first_key == "embedding.weight" else 1]
    num_classes = state["fc.weight"].shape[0]
    _, model_class, params = VARIANTS[variant]
    model = model_class(vocab_size, num_classes=num_classes, **params)
    model.load_state_dict(state)
    model.eval()

    x = torch.randint(1, vocab_size, (BATCH, LENGTH))
    before = peak_rss_mb()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        started = time.perf_counter()
        for _ in range(REPS):
            model(x)
        elapsed = time.perf_counter() - started
    return {
        "params": count_parameters(model),
        "latency_ms": elapsed * 1000 / (REPS * BATCH),
        "peak_rss_increase_mb": peak_rss_mb() - before,
        "peak_rss_mb": peak_rss_mb(),
    }


def hardware() -> dict:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {
        "cpu": cpu,
        "os": platform.platform(),
        "threads": int(os.environ.get("TORCH_THREADS", "6")),
        "torch": torch.__version__,
        "batch": BATCH,
        "length": LENGTH,
    }


def run_all():
    costs = {"hardware": hardware(), "models": {}}
    for variant in FAMILIES:
        runs = {}
        for seed in SEEDS:
            out = subprocess.check_output([sys.executable, __file__, variant, str(seed)], text=True)
            runs[str(seed)] = json.loads(out.strip().splitlines()[-1])
            print(variant, seed, runs[str(seed)], flush=True)
        costs["models"][variant] = runs
    with open("artifacts/costs.json", "w") as f:
        json.dump(costs, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(measure_one(sys.argv[1], int(sys.argv[2]))))
    else:
        run_all()
