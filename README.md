# alpaca-segmentation 🦙🎧

**Utilities for dataset preparation, feature extraction, Random-Forest (RF) training, and analysis**, used alongside the [`ANIMAL-SPOT-alpaca`](#coupling-with-the-fork) pipeline for **CNN training/inference/evaluation on HPC**.
This repo is the **“data + post-processing + RF”** side of the project.

---

## Table of Contents

* [Overview](#overview)
* [What Lives Here vs. Fork](#what-lives-here-vs-fork)
* [Setup](#setup)
* [Workflows](#typical-workflows)
* [Conventions & Defaults](#conventions--defaults)
* [Results](#current-results-brief)
* [Troubleshooting](#troubleshooting-greatest-hits)
* [Coupling With the Fork](#coupling-with-the-fork)
* [Roadmap](#roadmap--todo)

---

## Overview

This repository complements the [`ANIMAL-SPOT-alpaca`](https://github.com/darizae/ANIMAL-SPOT-alpaca) fork by focusing on:

* **Dataset preparation**
* **Python-based audio feature extraction**
* **Random-Forest (RF) model training & evaluation**
* **Metrics analysis & result summarization**

---

## What Lives Here vs. Fork

**Here:**

* Dataset prep scripts
* Selection tables
* **Python audio features**
* RF training & summaries
* Evaluation scripts & notebooks
* Pulled run artefacts (`TRAINING/runs/**` summaries)

**In [`ANIMAL-SPOT-alpaca`](#coupling-with-the-fork):**

* CNN configs
* Slurm factories
* GPU predictions
* CPU evaluation
* RF inference layer for benchmark runs

**Key entry points:**

* `data_postprocessing/evaluate_benchmark.py`
* `random_forest/train_random_forest.py`
* `random_forest/metrics/build_metrics_summary.py`
* `random_forest/extract_audio_features.py`

---

## Setup

```bash
# Python ≥3.11 recommended
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Typical Workflows

### **A) Prepare / Inspect Datasets**

```bash
python data_preprocessing/build_alpaca_index.py
python data_preprocessing/make_selection_tables.py
python data_preprocessing/count_dataset_files.py
```

### **B) Pull Results from HPC**

```bash
bash data_postprocessing/pull_runs.sh [optional/subtree]
```

### **C) Score Benchmark Runs**

```bash
python data_postprocessing/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs ../ANIMAL-SPOT-alpaca/BENCHMARK/runs \
  --iou 0.40 --layer both \
  --out data_postprocessing/BENCHMARK/metrics.csv \
  --per-tape-out data_postprocessing/BENCHMARK/metrics_per_tape.csv
```

### **D) Train RF Classifiers**

```bash
python random_forest/train_random_forest.py
python random_forest/metrics/build_metrics_summary.py
```

### **E) Feature Extraction for Deployment**

```bash
python random_forest/extract_audio_features.py \
  --wav PATH/TO/*.wav_0_####.wav \
  --selections PATH/TO/selection_table.txt \
  --out features.csv \
  [--include-deltas]
```

---

## Conventions & Defaults

* **Evaluation unit:** hum segment; IoU=0.40
* **CNN threshold:** 0.90
* **RF threshold:** 0.70
* **Labels:** selection tables use `Sound type ∈ {target, noise}`
* **File naming:** utilities normalize tape names for joins

---

## Current Results (Brief)

* **RF training:** best F1 **0.885** (manual features, 2:1)
* **Python + CNN-logit:** ≈ **0.876**, ROC-AUC ≈ **0.97**
* **Pipeline effect (13 tapes, 4 models):**
  CNN=0.90, RF=0.70 → **ΔPrecision +0.23**, **ΔRecall −0.08**, **ΔF1 +0.258**

---

## Troubleshooting (Greatest Hits)

* **“WAV not found”** → Normalize tape names; handle suffixes.
* **Gaps / drift in selection numbers** → Use *Raven → Renumber Selections*.
* **Unlabeled rectangles** → Fill `Sound type` column manually.
* **Filename mismatch** → Keep `.txt` and `.wav` basenames identical.

---

## Coupling With the Fork

This repo expects outputs from **[`ANIMAL-SPOT-alpaca`](https://github.com/darizae/ANIMAL-SPOT-alpaca)**:

* CNN evaluation selection tables + `evaluation/index.json`
* If using the RF post-filter in the fork, it writes `postrf/index.json`
* This repo’s `evaluate_benchmark.py` consumes both (`--layer both`)

---

## Roadmap / TODO

* [ ] Building and publishing a comprehensive dataset.
