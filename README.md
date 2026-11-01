# AIGC 5500 Midterm Project: Comparing Convolutional, Recurrent, and Attention-Based Architectures

**Course:** AIGC 5500, Advanced Deep Learning
**Group:** Group 01: Ranjay Salmon, Ankit Yadav, Ripan Deep Kour, Teja Ekkurthi
**Primary dataset:** SNIPS voice-assistant intents (7 classes)
**Second dataset:** ATIS flight-booking intents (16 classes after filtering)

## Question

Given the same split, optimizer, training budget and tuning effort, how do a 1D CNN, a Bi-LSTM and a Transformer encoder compare at short-text intent classification, and does a bag-of-words baseline already do the job? We also check which component of each model matters, whether the findings hold on a second dataset, and what each model costs to run.

## Datasets

Both download automatically from Hugging Face when the code runs:

- SNIPS: `load_dataset("benayas/snips")`
- ATIS: `load_dataset("tuetschek/atis")`

Repeated utterances are removed before splitting, so no text appears in more than one split. Classes need at least 10 unique examples. The split is stratified 70/15/15 with a fixed seed.

| Dataset | Train | Validation | Test | Classes | Vocabulary |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SNIPS | 9,983 | 2,139 | 2,140 | 7 | 3,671 |
| ATIS | 3,819 | 819 | 819 | 16 | 590 |

## Setup

Choose the requirements file for your machine. Both are kept so developers on different setups can each have their own environment:

- `requirements-intel-mac.txt`: Intel Macs, tested with Python 3.12 and PyTorch 2.2.2. These are the versions used to run `cost.py`, the notebook and the report build.
- `requirements.txt`: the original pins (PyTorch 2.15.0) for machines with builds of that version. It has no build for Intel Macs, and we haven't rerun this project on it.

```bash
git clone https://github.com/Yadavank02/Midterm_Project_DeepLearning_Group01.git
cd Midterm_Project_DeepLearning_Group01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-intel-mac.txt
```

## How to run

Open `midterm.ipynb` and run all cells. With `RUN_EXPERIMENTS = False` it reproduces every table and figure from the committed results. With `True` it retrains everything first. The same steps from the command line:

```bash
python src/main.py --fresh
python src/cost.py
python src/compare.py
python src/diagrams.py
```

`--fresh` ignores the committed results and retrains everything, which overwrites `artifacts/results.json` and `artifacts/predictions.json`. `main.py` saves after every run, and running it without `--fresh` resumes an interrupted run. Trained weights go to `checkpoints/`, which is not committed, so `cost.py` needs `main.py` to have been run on that machine first.

**Seeds:** 42, 123, 999. The split seed is fixed at 42, and seeds vary initialisation and shuffling.

**Hardware:** training ran on an Intel Core i9-9980HK CPU and a Colab Tesla T4 GPU, recorded for each run in `artifacts/results.json` (runs with no device field ran on the CPU). Cost was measured on the i9-9980HK with 6 threads.

## Results

Test macro F1 for every seed. Each family tuned the same 10-setting grid on validation at seed 42.

### SNIPS

| Model | Seed 42 | Seed 123 | Seed 999 | Mean | Spread |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9812 | 0.9822 | 0.9827 | 0.9820 | 0.0014 |
| CNN | 0.9842 | 0.9822 | 0.9814 | 0.9826 | 0.0028 |
| Bi-LSTM | 0.9807 | 0.9803 | 0.9818 | 0.9809 | 0.0015 |
| Transformer | 0.9775 | 0.9738 | 0.9765 | 0.9759 | 0.0037 |

### Ablations (SNIPS)

| Component removed | Standard | Ablated | Change | Per seed (42, 123, 999) |
| :--- | ---: | ---: | ---: | :--- |
| CNN: BatchNorm | 0.9826 | 0.9803 | -0.0023 | -0.0014, -0.0027, -0.0028 |
| LSTM: backward pass | 0.9809 | 0.9789 | -0.0021 | -0.0004, +0.0000, -0.0058 |
| Transformer: positions | 0.9759 | 0.9769 | +0.0010 | +0.0014, +0.0028, -0.0013 |

### ATIS

| Model | Seed 42 | Seed 123 | Seed 999 | Mean | Spread |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.7649 | 0.7602 | 0.7277 | 0.7509 | 0.0371 |
| CNN | 0.8462 | 0.8248 | 0.8387 | 0.8366 | 0.0214 |
| Bi-LSTM | 0.8343 | 0.8271 | 0.8483 | 0.8365 | 0.0212 |
| Transformer | 0.8019 | 0.7370 | 0.7765 | 0.7718 | 0.0649 |

### Inference cost

| Model | Parameters | ms per sample | Peak process memory (MB) | Increase during inference (MB) |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 25,704 | 0.0035 | 333.9 | 0.4 |
| CNN | 588,095 | 0.0566 | 342.6 | 7.6 |
| Bi-LSTM | 735,879 | 0.1118 | 351.5 | 15.4 |
| Transformer | 735,751 | 0.1272 | 355.0 | 17.3 |

### What we found

- On SNIPS, the baseline and CNN can't be separated, and every Transformer seed falls below every other model's seeds.
- On ATIS, every CNN and Bi-LSTM seed beats every baseline and Transformer seed. The two tie, and the Transformer overlaps the baseline.
- No ablation shows a large effect on SNIPS. BatchNorm removal lowers all three seeds slightly, but that comparison also crosses devices.
- Explanations of why the models differ are our interpretation, not tested causes. See section 8 of the report for limitations, including mixed training hardware.

## Repository layout

```text
src/data_pipeline.py   loading, duplicate removal, splits, vocabulary, DataLoaders
src/models.py          baseline, CNN, Bi-LSTM, Transformer with ablation toggles
src/main.py            tuning and all experiment runs
src/cost.py            latency and peak memory from trained checkpoints
src/compare.py         tables, figures and confusion matrices
src/diagrams.py        architecture diagram traced from models.py
midterm.ipynb          runnable notebook
report/report.pdf      written report
slides/slides.pdf      presentation slides
artifacts/             results, predictions, costs and figures
```

## Video presentation

* **Link:** https://humberital-my.sharepoint.com/:v:/g/personal/n01770622_humber_ca/IQC9aGyGlsPBT611KSSxdbQ_AeX2yb5FMczYU1oyGJNB808
* **Presenter:** Teja Ekkurthi
* **Length:** 19min 11sec

## Contributions

| Member | Humber Student ID | Piece | Ablation owned |
| :--- | :--- | :--- | :--- |
| Ranjay Salmon | N10039663 | Project lead, baseline and shared evaluation | None |
| Ankit Yadav | N10040403 | Convolutional model | BatchNorm |
| Ripan Deep Kour | N01454601 | Recurrent model | Bidirectionality |
| Teja Ekkurthi | N01770622 | Attention model and ATIS check | Positional encoding |

The commit history shows who committed each change, including the later fixes and reruns on `assignment-fixes`.

## References and AI assistance

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
3. Coucke, A., et al. (2018). *Snips Voice Platform*. arXiv:1805.10190.
4. Hemphill, C. T., et al. (1990). *The ATIS Spoken Language Systems Pilot Corpus*. DARPA Speech and Natural Language Workshop.
5. Claude Code (Anthropic) helped review the code, implement fixes, generate figures, and draft the report and slides.

## References and Acknowledgements

### Course Materials
- Code examples and implementation techniques were adapted from previous AIGC5500 Advanced Deep Learning laboratory exercises.
- Lecture notes, starter code, and code files provided by the course instructor were used as references for model architecture, training, and evaluation.

### AI-Assisted Development
- AI tools, including ChatGPT, were used to support brainstorming, debugging, and troubleshooting during project development.
- AI assistance was used to help understand errors, explore possible solutions, and improve code documentation.
- The team reviewed and tested the final implementation and experimental results.

### Datasets
- SNIPS Intent Classification Dataset.
- ATIS (Airline Travel Information Systems) Dataset.
