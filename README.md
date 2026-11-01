# AIGC 5500 Midterm Project: Comparing Convolutional, Recurrent, and Attention-Based Architectures

**Course:** AIGC 5500 — Advanced Deep Learning  
**Project:** Midterm Project: Architecture Comparison on Short Text Classification  
**Group:** Group 01  
**Primary Dataset:** SNIPS Voice Assistant Intent Dataset (7 Classes)  
**Secondary Dataset:** ATIS Airline Travel Information System (Flight Booking Intents)  

---

## 1. Research Question

> **How do the inductive biases of Convolutional (1D CNN), Recurrent (Bi-LSTM), and Attention-Based (Transformer Encoder) architectures affect classification performance, convergence, and inference efficiency on short text intent classification when trained strictly from scratch under identical budgets?**

We evaluate all three deep neural network families against a defensible unigram Bag-of-Words Logistic Regression baseline across three random seeds (`42`, `123`, `999`), perform one controlled ablation per family, and test whether architectural rankings hold when transitioning from multi-domain voice assistant queries (SNIPS) to specialized flight-booking requests (ATIS).

---

## 2. Datasets & Access

1. **Primary Dataset: SNIPS Voice Assistant**
   * **Domain:** 7 balanced virtual assistant intent classes (*AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent*).
   * **Size:** ~14,484 samples partitioned into 70% Train, 15% Validation, and 15% Test.
   * **Sequence Length:** Short command utterances (~9 tokens on average, max length capped at 35).
   * **Source:** Extracted dynamically via `load_dataset("benayas/snips")`.

2. **Secondary Dataset: ATIS (Airline Travel Information System)**
   * **Domain:** Specialized airline travel and flight reservation queries with class imbalance.
   * **Size:** ~5,871 samples partitioned using identical 70/15/15 stratified splits.
   * **Source:** Extracted dynamically via `load_dataset("tuetschek/atis")`.

---

## 3. Project Architecture & File Structure

```text
Midterm_Project_DeepLearning_Group01/
├── data_pipeline.py      # Data extraction, EDA generation, tokenization, train-only vocab, DataLoaders
├── models.py             # PyTorch models: 1D CNN, Bi-LSTM, Transformer Encoder with ablation toggles
├── main.py               # Master experiment orchestrator: runs all 33 seed/ablation runs across seeds
├── eval_baseline.py      # Baseline evaluation, validation C-tuning & confusion matrix
├── eval_cnn.py           # CNN evaluation, BatchNorm ablation metrics & confusion matrix
├── eval_rnn.py           # Recurrent (Bi-LSTM) evaluation, bidirectionality ablation & confusion matrix
├── eval_transformer.py   # Transformer evaluation, positional ablation & confusion matrix
├── compare.py            # Consolidated comparison tables & publication figures (Figures 1, 2, 3)
├── requirements.txt      # Pinned environment dependencies
└── artifacts/            # Generated outputs, JSON logs, confusion matrices, and figure plots
```

---

## 4. How to Reproduce All Results

### Step 1: Environment Setup
Clone the repository and install dependencies in an isolated virtual environment:

```bash
git clone https://github.com/your-username/Midterm_Project_DeepLearning_Group01.git
cd Midterm_Project_DeepLearning_Group01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Full Benchmark Pipeline
Execute the full benchmark suite across all seeds (`42`, `123`, `999`):

```bash
python main.py
```
*Trains the baseline, the three deep models, and all ablations on both SNIPS and ATIS, automatically exporting results to `artifacts/results.json`.*

### Step 3: Generate Report Tables & Figures
```bash
python compare.py
```
*Outputs the formatted tables for Sections 4, 5, 6, and 7 of the report and writes Figures 1, 2, and 3 to `artifacts/`.*

### Step 4: Generate Individual Confusion Matrices
```bash
python eval_baseline.py
python eval_cnn.py
python eval_rnn.py
python eval_transformer.py
```

---

## 5. Experimental Results (Exact Outputs)

### Core Comparison (Primary Dataset: SNIPS)
*Hyperparameter selection guided by validation performance; test set evaluated once.*

| Architecture | Seed 42 | Seed 123 | Seed 999 | Mean Macro F1 | Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Reg)** | 0.9707 | 0.9671 | 0.9699 | **0.9692** | 0.0036 |
| **Convolutional (1D CNN)** | 0.9848 | 0.9806 | 0.9839 | **0.9831** | 0.0042 |
| **Recurrent (Bi-LSTM)** | 0.9829 | 0.9811 | 0.9798 | **0.9813** | 0.0032 |
| **Attention (Transformer)** | 0.9829 | 0.9779 | 0.97560 | **0.9788** | 0.0074 |

### Ablation Studies (SNIPS)
*Each ablation isolates and modifies exactly one architectural component, holding all other hyperparameters fixed.*

| Family | Component Tested | Standard Macro F1 | Ablated Macro F1 | Delta ($\Delta$) | Empirical Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Convolutional** | Batch Normalization | 0.9831 | 0.9842 | **+0.0011** | Removing BatchNorm showed minimal variance on SNIPS due to short, consistent sequence contexts. |
| **Recurrent** | Bidirectionality | 0.9813 | 0.9799 | **-0.0014** | Unidirectional processing suffers a drop because future contextual tokens help disambiguate early command verbs. |
| **Attention** | Positional Encoding | 0.9788 | 0.9791 | **+0.0003** | Positional encodings had a negligible impact, indicating local keyword presence dominates over strict token ordering in SNIPS. |

### Second Dataset Generalization Check (ATIS)
*Evaluating whether the ranking and inductive biases observed on SNIPS generalize to ATIS.*

| Architecture | SNIPS Mean F1 | ATIS Mean F1 | ATIS Rank | Architectural Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Logistic Reg)** | 0.9692 | 0.7162 | Rank 4 | Bag-of-words macro F1 collapses significantly on ATIS due to severe class imbalance and overlapping vocabulary. |
| **Convolutional (1D CNN)** | 0.9831 | 0.7968 | Rank 3 | Captures local phrasal triggers (*"flight from"*, *"round trip"*), improving over baseline. |
| **Recurrent (Bi-LSTM)** | 0.9813 | 0.8161 | Rank 1 | Masked pooling over temporal steps effectively resolves complex sequential syntax in flight requests. |
| **Attention (Transformer)** | 0.9788 | 0.8132 | Rank 2 | Self-attention provides strong generalization on ATIS by directly linking source and destination slots. |

> **Conclusion on Second Dataset:** **Ranking Reversal Confirmed.** While keyword-dense SNIPS allows simpler local/BoW features to perform competitively, the syntactic complexity and slot dependencies of ATIS cause the architectural ranking to shift, with deep sequence models—led by Bi-LSTM (0.8161) and the Transformer (0.8132)—vastly outperforming the baseline macro F1 (0.1762).

### Operational Inference Cost Profiling
*Measured on test batches with identical sequence length ($L=35$ tokens).*

| Architecture | Total Trainable Parameters | Inference Latency / Sample (ms) | Peak Memory Allocation |
| :--- | :---: | :---: | :--- |
| **Convolutional (1D CNN)** | 587,071 | 0.1187 ms | Minimal (CPU/MPS BLAS) |
| **Recurrent (Bi-LSTM)** | 734,855 | 0.1836 ms | Minimal (CPU/MPS BLAS) |
| **Attention (Transformer)** | 734,727 | 0.2523 ms | Minimal (CPU/MPS BLAS) |

*Parameter Counts:* All three deep architectures are tightly matched within the required parameter bounds.

---

## 6. Video Presentation

* **Link:** 
* **Presenter:** 
* **Duration:** 
* **Access Permissions:** Humber-wide viewable link + directly shared with `hossein.pourmodheji@humber.ca`.

---

## 7. Group Contribution Table

| Member Name | Humber Student ID | Assigned Piece & Role | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Ranjay Salmon** |  | Project Lead & Baseline | Orchestrating main pipeline execution, data preprocessing pipeline, stratified splitting, Logistic Regression baseline tuning, and inference profiling |
| **Ankit Yadav** | N10040403 | Convolutional Model (1D CNN) | TextCNN implementation (`models.py`), Batch Normalization ablation study, convolutional receptive field analysis, and confusion matrix generation |
| **Ripan Deep Kour** | N01454601 | Recurrent Model (LSTM) | Bi-LSTM implementation (`models.py`), masked temporal pooling mechanism, bidirectionality ablation study, and training loss curve convergence tracking |
| **Teja Ekkurthi** |  | Attention Model & ATIS | Transformer Encoder implementation, positional encoding ablation, ATIS secondary dataset porting, cross-dataset ranking analysis, and final report consolidation |

---

## 8. Academic Integrity & References

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP 2014.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. Coucke, A., et al. (2018). *Snips Voice Platform: An embedded Spoken Language Understanding system*. arXiv:1805.10190.
4. Hemphill, C. T., et al. (1990). *The ATIS spoken language systems pilot corpus*. DARPA Speech and Natural Language Workshop.
