"""
build_report.py
Builds report/report.pdf. Tables and quoted results are read from artifacts/*.json, so the report
can't disagree with the results. Run from the repository root.
"""

import json

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEEDS = ["42", "123", "999"]
CORE = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]
NAMES = {
    "Baseline": "Baseline", "CNN_Standard": "CNN", "CNN_NoBN": "CNN, no BatchNorm",
    "RNN_BiLSTM": "Bi-LSTM", "RNN_UniLSTM": "LSTM, one direction",
    "Transformer_Standard": "Transformer", "Transformer_NoPos": "Transformer, no positions",
}

results = json.load(open("artifacts/results.json"))
costs = json.load(open("artifacts/costs.json"))
snips, atis = results["SNIPS"], results["ATIS"]


def f1s(data, model):
    return [data["variants"][model]["seeds"][s]["test_f1"] for s in SEEDS]


def mean(data, model):
    return float(np.mean(f1s(data, model)))


def seed_table(data, models):
    rows = [["Model", "Seed 42", "Seed 123", "Seed 999", "Mean", "Spread"]]
    for m in models:
        v = f1s(data, m)
        rows.append([NAMES[m]] + [f"{x:.4f}" for x in v] + [f"{np.mean(v):.4f}", f"{np.ptp(v):.4f}"])
    return rows


styles = getSampleStyleSheet()
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10, leading=12.6, spaceAfter=5)
H1 = ParagraphStyle("H", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor("#14213D"))
TITLE = ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, spaceAfter=3,
                       textColor=colors.HexColor("#14213D"))
SUB = ParagraphStyle("S", parent=BODY, alignment=1, spaceAfter=8)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8.5, leading=10, textColor=colors.HexColor("#46536B"))


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#14213D")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def figure(name, width, ratio):
    return Image(f"artifacts/{name}.png", width=width * inch, height=width * ratio * inch)


s = []
P = lambda text: s.append(Paragraph(text, BODY))
H = lambda text: s.append(Paragraph(text, H1))
GAP = lambda: s.append(Spacer(1, 6))

s.append(Paragraph("CNN, LSTM and Transformer on Short-Text Intent Classification", TITLE))
s.append(Paragraph("AIGC 5500 Advanced Deep Learning, Midterm Project, Group 01: RJ and Ankit Yadav", SUB))

H("1 Problem and data")
m_s, m_a = snips["meta"], atis["meta"]
P("We asked how convolutional, recurrent and attention-based models compare at classifying the intent of a short "
  "utterance when they share the same split, optimizer, training budget and tuning effort, and whether a bag-of-words "
  "baseline already does the job. SNIPS is our primary dataset: voice-assistant commands in "
  f"{m_s['num_classes']} balanced intents. ATIS is our second dataset: flight-booking queries in {m_a['num_classes']} "
  "intents, where the “flight” class alone is 606 of the 819 test rows and the rarest class has one test row.")
s.append(table([
    ["Dataset", "Train", "Validation", "Test", "Classes", "Vocabulary"],
    ["SNIPS", m_s["split_sizes"]["train"], m_s["split_sizes"]["val"], m_s["split_sizes"]["test"], m_s["num_classes"], m_s["vocab_size"]],
    ["ATIS", m_a["split_sizes"]["train"], m_a["split_sizes"]["val"], m_a["split_sizes"]["test"], m_a["num_classes"], m_a["vocab_size"]],
]))
GAP()
P("Our first pipeline split the raw data. When we checked, about 2% of SNIPS test rows and 8–9% of ATIS test rows also "
  "appeared word for word in training, which is leakage. We now remove repeated utterances before a stratified 70/15/15 "
  "split, confirmed zero overlap between splits, and drop classes with fewer than 10 unique examples so that every split "
  "contains each class. The split seed is fixed, so seeds vary only model initialisation and shuffling. We lowercase, "
  "tokenize by regex, build the vocabulary from training text only, and pad or cut to 35 tokens.")

H("2 The three models and why")
P("Each model starts with a 128-dimension embedding trained from scratch. <b>TextCNN</b> runs convolutions over windows of "
  "2, 3 and 4 words, so it responds to local phrases wherever they occur. <b>TextRNN</b> is a one-layer bidirectional LSTM "
  "with masked mean pooling, so it builds state along the sequence. <b>TextTransformer</b> is a two-layer, four-head encoder "
  "with sinusoidal positions and padding masks, so any word can look at any other directly. Parameter counts are "
  f"{snips['variants']['CNN_Standard']['seeds']['42']['params']:,}, {snips['variants']['RNN_BiLSTM']['seeds']['42']['params']:,} and "
  f"{snips['variants']['Transformer_Standard']['seeds']['42']['params']:,} on SNIPS, within a factor of 1.3.")
s.append(figure("figure0_architectures", 6.9, 6.2 / 13))
s.append(Paragraph("Figure 1. Layer shapes recorded by running a batch through the classes in models.py (diagrams.py).", CAP))

H("3 Baseline")
P("Our baseline is logistic regression on log-scaled word counts. It reads the same token ids as the deep models and shares "
  f"their optimizer, 20-epoch budget and tuning. It has only {snips['variants']['Baseline']['seeds']['42']['params']:,} "
  "parameters on SNIPS, far below the deep models, because being a much simpler model is its purpose. Every family tuned the "
  "same grid of 10 settings: learning rates 0.0003, 0.001, 0.003, 0.01 and 0.03, each with weight decay 0 and 0.01, at seed 42, "
  "chosen by validation macro F1. Ablations keep their family’s settings.")
rows = [["Family", "SNIPS lr, decay", "SNIPS val F1", "ATIS lr, decay", "ATIS val F1"]]
for m in CORE:
    a, b = max(snips["tuning"][m], key=lambda r: r["val_f1"]), max(atis["tuning"][m], key=lambda r: r["val_f1"])
    rows.append([NAMES[m], f"{a['lr']}, {a['weight_decay']}", f"{a['val_f1']:.4f}", f"{b['lr']}, {b['weight_decay']}", f"{b['val_f1']:.4f}"])
s.append(table(rows))
GAP()
P("Our first grid stopped at a learning rate of 0.001. The baseline then underfit, with ATIS validation F1 of 0.21 and test F1 "
  "of 0.18, and that learning rate was the best available for every family, so we widened the grid for all of them. The baseline’s "
  "best setting is still the top of the grid, so it might gain a little from going higher.")

H("4 Results")
s.append(Paragraph("Table: SNIPS test macro F1, every seed.", CAP))
s.append(table(seed_table(snips, CORE)))
GAP()
P(f"On SNIPS the CNN ({mean(snips, 'CNN_Standard'):.4f}) and baseline ({mean(snips, 'Baseline'):.4f}) can’t be separated: their seeds "
  f"overlap. The Bi-LSTM ({mean(snips, 'RNN_BiLSTM'):.4f}) is close behind. All three Transformer seeds fall below every seed of the other "
  "models. Our explanation is that intents here are signalled by keywords such as “play” or “weather”, which a word-count "
  "model or a local filter finds directly. In the Transformer’s seed-42 confusion matrix, the largest group of errors is SearchCreativeWork rows predicted as "
  "PlayMusic or SearchScreeningEvent, classes that can share titles and names. This mechanism is our interpretation; we did not test it directly.")
s.append(figure("figure2_dataset_generalization", 6.5, 900 / 2200))
s.append(Paragraph("Figure 2. Test macro F1 on both datasets; dots are seeds and bars are means. The y-axes differ.", CAP))
s.append(figure("figure1_loss_convergence", 6.5, 900 / 2200))
s.append(Paragraph("Figure 3. Training loss at seed 42. The Transformer falls most slowly on both datasets.", CAP))

H("5 Ablations")
s.append(Paragraph("Table: SNIPS test macro F1 with one component removed, every seed.", CAP))
s.append(table(seed_table(snips, ["CNN_NoBN", "RNN_UniLSTM", "Transformer_NoPos"])))
GAP()
rows = [["Component removed", "Standard", "Ablated", "Change", "Per seed (42, 123, 999)"]]
for label, a, b in [("CNN: BatchNorm", "CNN_Standard", "CNN_NoBN"), ("LSTM: backward pass", "RNN_BiLSTM", "RNN_UniLSTM"),
                    ("Transformer: positions", "Transformer_Standard", "Transformer_NoPos")]:
    x, y = f1s(snips, a), f1s(snips, b)
    rows.append([label, f"{np.mean(x):.4f}", f"{np.mean(y):.4f}", f"{np.mean(y) - np.mean(x):+.4f}",
                 ", ".join(f"{q - p:+.4f}" for p, q in zip(x, y))])
s.append(table(rows))
GAP()
P("Removing BatchNorm lowered the CNN on all three seeds, a small but consistent change, although the standard CNN ran on the CPU and the "
  "ablated CNN on the GPU, so the device differs as well as the component. Removing the LSTM’s backward pass made almost no "
  "difference on two seeds and a larger one on the third, so we can’t call it an effect; this ablation also removes about 133,000 parameters, "
  "so it changes capacity as well as direction. Removing positions did not hurt the Transformer on average, which fits keyword-driven intents "
  "where word order adds little. With three seeds we treat all of these as indications only.")

H("6 Second dataset")
s.append(Paragraph("Table: ATIS test macro F1, every seed.", CAP))
s.append(table(seed_table(atis, CORE)))
GAP()
P("We chose ATIS because it is a similar task with far stronger class imbalance. The ranking changes. The CNN and Bi-LSTM tie, with all "
  "their seeds above every baseline and Transformer seed. That gap is the clearest result in this project: on harder, imbalanced data, "
  "sequence-aware models beat the bag of words, which they couldn’t do on SNIPS. The Transformer does not share that advantage: it has the "
  "widest spread and overlaps the baseline. ATIS has far less training data than SNIPS, which may be part of the reason, but we did not test that. "
  "A caution on macro F1: rare classes with a handful of test rows carry the same weight as “flight”, and at seed 42 all four models scored "
  "zero F1 on “meal”, which has one test row.")

H("7 Inference cost")
hw = costs["hardware"]
rows = [["Model", "Parameters", "ms per sample", "Peak process memory (MB)", "Increase during inference (MB)"]]
for m in CORE:
    r = list(costs["models"][m].values())
    rows.append([NAMES[m], f"{r[0]['params']:,}", f"{np.mean([x['latency_ms'] for x in r]):.4f}",
                 f"{np.mean([x['peak_rss_mb'] for x in r]):.1f}", f"{np.mean([x['peak_rss_increase_mb'] for x in r]):.1f}"])
s.append(table(rows))
GAP()
P(f"Measured on {hw['cpu']}, {hw['threads']} threads, PyTorch {hw['torch']}, batches of {hw['batch']} at length {hw['length']}, using the "
  "trained SNIPS checkpoints for all three seeds, with each measurement in a fresh process. Process memory includes the Python and PyTorch runtime, "
  "so the increase column compares the models better. The baseline is about 16 times faster than the CNN at a comparable SNIPS score, so on "
  "SNIPS the simple model is the sensible choice. The CNN costs about half as much time as the Bi-LSTM with the same ATIS score.")

H("8 Limitations")
P("(1) Three seeds on one split can’t support significance claims. (2) Tuning used one seed and the grid is coarse. (3) Test scores from our "
  "earlier runs were visible while we were revising the pipeline, including the baseline’s 0.18. Validation scores drove the final choice of "
  "settings and the validation underfit was visible on its own, but the final test set wasn’t untouched by us in the strictest sense. (4) Training ran on two "
  "devices: a Mac CPU and a Colab Tesla T4, recorded per run in results.json. All Transformer runs and the SNIPS baseline and standard CNN were on the CPU; the other final runs were on the T4. "
  "So some comparisons cross devices, which we would remove by rerunning everything on one. Cost was measured on one machine. (5) ATIS rare classes have very few test rows, so macro F1 there moves with a few predictions. "
  "(6) The explanations of why models differ are interpretations, not tested causes.")

H("9 Contributions and references")
P("<b>Members:</b> RJ and Ankit Yadav. Per the commit history, Ankit wrote the original pipeline, models and first runs, and RJ made the later fixes "
  "and reran the experiments. <b>Ablation owners:</b> [TO COMPLETE: name the owner of the CNN, LSTM and Transformer ablations].")
P("<b>AI assistance:</b> Claude Code (Anthropic) helped review the code, implement the fixes, generate figures and draft this report. We reviewed the results. "
  "<b>References:</b> Kim (2014), Convolutional Neural Networks for Sentence Classification, EMNLP. Vaswani et al. (2017), Attention Is All You Need, NeurIPS. "
  "Coucke et al. (2018), Snips Voice Platform, arXiv:1805.10190. Hemphill et al. (1990), The ATIS Spoken Language Systems Pilot Corpus, DARPA Workshop. "
  "Hugging Face datasets benayas/snips and tuetschek/atis.")

SimpleDocTemplate("report/report.pdf", pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                  topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="AIGC 5500 Midterm Report, Group 01").build(s)
