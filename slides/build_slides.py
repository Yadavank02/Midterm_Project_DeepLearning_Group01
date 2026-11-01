"""
build_slides.py
Builds slides/slides.pdf: problem, method, data, process, each round of runs, and what we identified.
Final-round tables come from artifacts/*.json. Earlier rounds are fixed records of runs since replaced.
Run from the repository root.
"""

import json

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

W, H = 1280, 720
M = 72
DARK, LIGHT, ACCENT, MUTED = HexColor("#14213D"), HexColor("#F7F6F1"), HexColor("#C2410C"), HexColor("#46536B")
SEEDS = ["42", "123", "999"]
CORE = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]
NAMES = {"Baseline": "Baseline", "CNN_Standard": "1D CNN", "RNN_BiLSTM": "Bi-LSTM", "Transformer_Standard": "Transformer"}

results = json.load(open("artifacts/results.json"))
costs = json.load(open("artifacts/costs.json"))
c = canvas.Canvas("slides/slides.pdf", pagesize=(W, H))
c.setTitle("AIGC 5500 Midterm Slides, Group 01")


def f1s(dataset, model):
    return [results[dataset]["variants"][model]["seeds"][s]["test_f1"] for s in SEEDS]


def text(body, y, size=26, color=DARK, width=W - 2 * M, x=M, bold=False):
    """Draws wrapped text with its top at y and returns the y below it."""
    style = ParagraphStyle("p", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=size * 1.28, textColor=color)
    para = Paragraph(body, style)
    _, used = para.wrap(width, H)
    para.drawOn(c, x, y - used)
    return y - used - size * 0.6


def page(title, kicker=None, dark=False):
    c.setFillColor(DARK if dark else LIGHT)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    y = H - 60
    if kicker:
        y = text(kicker, y, 20, HexColor("#F4A261") if dark else ACCENT)
    return text(title, y, 44, LIGHT if dark else DARK, bold=True)


def table(rows, y, widths, size=23, color=DARK):
    t = Table(rows, colWidths=widths, rowHeights=size * 1.75)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", size),
        ("FONT", (0, 1), (-1, -1), "Helvetica", size),
        ("TEXTCOLOR", (0, 0), (-1, -1), color),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.6, HexColor("#9AA3B2")),
    ]))
    _, used = t.wrap(W, H)
    t.drawOn(c, M, y - used)
    return y - used - 26


def image(name, y, height):
    reader = ImageReader(f"artifacts/{name}.png")
    w, h = reader.getSize()
    width = min(W - 2 * M, height * w / h)
    height = width * h / w
    c.drawImage(reader, (W - width) / 2, y - height, width, height)
    return y - height - 16


def bullets(items, y, size=28):
    for item in items:
        y = text(f"&bull;&nbsp; {item}", y, size)
    return y


def seed_rows(dataset):
    rows = [[dataset, "Seed 42", "Seed 123", "Seed 999", "Mean", "Spread"]]
    for m in CORE:
        v = f1s(dataset, m)
        rows.append([NAMES[m]] + [f"{x:.4f}" for x in v] + [f"{np.mean(v):.4f}", f"{np.ptp(v):.4f}"])
    return rows


# Cover
c.setFillColor(DARK)
c.rect(0, 0, W, H, stroke=0, fill=1)
y = text("AIGC 5500 Midterm, Group 01: Ranjay Salmon, Ankit Yadav, Ripan Deep Kour, Teja Ekkurthi", H - 190, 24, HexColor("#F4A261"))
y = text("Which architecture best reads a short request?", y, 64, LIGHT, bold=True)
text("Comparing a CNN, an LSTM and a Transformer against a simple baseline", y, 30, HexColor("#C9D3E3"))
c.showPage()

y = page("Our problem")
y = text("When someone types a short request, can we tell what they want, and which kind of model does that best?", y, 38, bold=True)
text("We tried to identify how much the architecture matters when every model gets the same data, budget and tuning, and whether a simple model already does the job.", y, 30, MUTED)
c.showPage()

y = page("To identify this, we used four models")
y = bullets(["<b>Baseline:</b> logistic regression on word counts, which ignores word order",
             "<b>1D CNN:</b> looks for local patterns of 2 to 4 words",
             "<b>Bi-LSTM:</b> reads the sentence in both directions, keeping state",
             "<b>Transformer:</b> lets every word look directly at every other word"], y)
text("Three seeds each, one component removed per family, then a second dataset.", y - 10, 28, ACCENT)
c.showPage()

y = page("The three deep models, from our code")
image("figure0_architectures", y, 545)
c.showPage()

y = page("The data contained two kinds of request")
y = text("<b>SNIPS:</b> voice-assistant commands, 7 balanced intents, 14,262 unique utterances. "
         "<b>ATIS:</b> flight queries, 16 intents, 5,457 unique utterances, about 74% “flight”.", y, 26)
image("atis_eda_distribution", y, 400)
c.showPage()

y = page("We ran it through one shared process")
y = bullets(["<b>Split:</b> stratified 70/15/15", "<b>Encode:</b> train-only vocabulary, 35 tokens",
             "<b>Train:</b> AdamW, 20 epochs, three seeds", "<b>Select:</b> best epoch on validation",
             "<b>Score:</b> test macro F1, reported for every seed"], y)
text("Macro F1 counts every intent equally, however rare.", y - 10, 28, ACCENT)
c.showPage()

y = page("Results of our first runs", "Round 1")
y = table([["Model", "SNIPS mean F1", "ATIS mean F1"], ["Baseline (TF-IDF)", "0.9837", "0.7426"], ["1D CNN", "0.9828", "0.7953"],
           ["Bi-LSTM", "0.9813", "0.8161"], ["Transformer", "0.9808", "0.8192"]], y, [460, 340, 340], 26)
text("Original pipeline, commit 9a647c8. Each seed also used a different data split.", y, 22, MUTED)
c.showPage()

y = page("The trends we saw first")
bullets(["<b>The simple model kept up on SNIPS:</b> all four sat near 0.98, with gaps smaller than seed variation",
         "<b>The Transformer looked best on ATIS,</b> but one seed reversed the deep models’ lead over the baseline",
         "<b>Removing components changed almost nothing:</b> all three moved F1 by less than the seed spread"], y, 30)
c.showPage()

y = page("What we found, and what we updated")
table([["What we found", "What we updated"],
       ["2% of SNIPS and 8–9% of ATIS test rows also in training", "Removed duplicates before splitting"],
       ["Only the baseline was tuned", "Same validation grid for every family"],
       ["Baseline used different inputs and training", "Same token inputs, optimizer and budget"],
       ["Each seed changed the data split", "One fixed split per dataset"],
       ["Deep confusion matrices came from untrained models", "Built from trained models’ predictions"],
       ["Peak memory was not measured", "Measured for all models in fresh processes"]], y, [640, 496], 21)
c.showPage()

y = page("The rerun exposed a new problem", "Round 2", dark=True)
y = table([["Model", "SNIPS mean F1", "ATIS mean F1"], ["Baseline", "0.9764", "0.1831"], ["1D CNN", "0.9791", "0.8132"],
           ["Bi-LSTM", "0.9783", "0.8037"], ["Transformer", "0.9759", "0.7718"]], y, [460, 340, 340], 26, LIGHT)
text("The baseline wasn’t bad at ATIS; it had underfit. So we widened the learning-rate grid for every family.", y, 28, HexColor("#C9D3E3"))
c.showPage()

y = page("Results for every seed", "Round 3: final")
y = table(seed_rows("SNIPS"), y, [300, 170, 170, 170, 165, 165], 20)
y = table(seed_rows("ATIS"), y, [300, 170, 170, 170, 165, 165], 20)
c.showPage()

y = page("The picture changed once the comparison was fair")
y = image("figure2_dataset_generalization", y, 440)
text("On ATIS, the Transformer dropped from first to last among the deep models. The CNN and Bi-LSTM clearly beat the baseline.", y, 24)
c.showPage()

y = page("Removing components changed little")
rows = [["Removed (SNIPS)", "Standard", "Ablated", "Change", "Per seed"]]
for label, a, b in [("CNN: BatchNorm", "CNN_Standard", "CNN_NoBN"), ("LSTM: backward pass", "RNN_BiLSTM", "RNN_UniLSTM"),
                    ("Transformer: positions", "Transformer_Standard", "Transformer_NoPos")]:
    x, z = f1s("SNIPS", a), f1s("SNIPS", b)
    rows.append([label, f"{np.mean(x):.4f}", f"{np.mean(z):.4f}", f"{np.mean(z) - np.mean(x):+.4f}", ", ".join(f"{q - p:+.4f}" for p, q in zip(x, z))])
y = table(rows, y, [340, 145, 145, 145, 361], 21)
y = text("Word order adding so little fits keyword-driven intents, though we didn’t test that cause directly.", y, 28)
text("The BatchNorm comparison also crosses devices: standard CNN on CPU, ablated CNN on GPU.", y, 22, MUTED)
c.showPage()

y = page("What each model costs to run")
rows = [["Model", "Parameters", "ms per sample", "Memory added (MB)"]]
for m in CORE:
    r = list(costs["models"][m].values())
    rows.append([NAMES[m], f"{r[0]['params']:,}", f"{np.mean([q['latency_ms'] for q in r]):.4f}", f"{np.mean([q['peak_rss_increase_mb'] for q in r]):.1f}"])
y = table(rows, y, [360, 250, 250, 276], 25)
y = text("The baseline runs about 16 times faster than the CNN at a comparable SNIPS score.", y, 28)
text(f"{costs['hardware']['cpu']}, {costs['hardware']['threads']} threads, batch 64, length 35, mean of three SNIPS checkpoints", y, 20, MUTED)
c.showPage()

y = page("What we identified")
y = text("The evidence supports", y, 28, HexColor("#1F5A8A"), bold=True)
y = bullets(["On easy, balanced text, a simple model is enough", "On harder, imbalanced text, the CNN and LSTM win",
             "The Transformer showed no advantage on either dataset", "Pipeline fairness changed the ranking"], y, 24)
y = text("We would not claim", y - 4, 28, ACCENT, bold=True)
bullets(["Statistical significance from three seeds", "That the CNN beats the LSTM on ATIS",
         "Tested causes for why the models differ", "Results free of device effects or earlier test exposure"], y, 24)
c.showPage()
c.save()
