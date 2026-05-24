# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multimodal Speech Emotion Recognition (SER) on the IEMOCAP corpus. Classifies utterances into **4 emotion classes** (Angry, Happy, Neutral, Sad). University course project — 4 sprints over 8 weeks.

## Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned for Google Colab free-tier compatibility — do not upgrade versions without verifying Colab compatibility.

## Running Notebooks

```bash
source .venv/bin/activate
jupyter notebook
```

## Data

Two sources covering the same 10,039 utterances:
- **`iemocap_full_dataset.csv`** — local metadata CSV, tracked in git
- **HuggingFace parquet** — ~1.4 GB, downloaded on first use, cached locally:

```python
from datasets import load_dataset
ds = load_dataset("AbstractTTS/IEMOCAP", split="train")
```

`data/` and `models/` directories are gitignored. Never commit dataset files, audio, parquet, or model weights.

## Critical Design Decisions

### Evaluation Protocol — LOSO (Leave-One-Session-Out)
IEMOCAP has 5 sessions, each with a unique speaker pair. Always split by session (hold out one full session as test, train on remaining four, rotate). **Random row splits are incorrect** — they allow the same speaker in both train and test, inflating scores artificially.

### Label Mapping (4-class)
| Original | 4-class |
|----------|---------|
| `ang` | angry |
| `hap` | happy |
| `exc` | happy (merged — heavily overlapping with happy) |
| `neu` | neutral |
| `sad` | sad |
| `xxx` | **dropped** (~25% of rows, no annotator agreement) |
| `fru`, `dis`, `fea`, `sur` | **dropped** (too few samples) |

### Primary Metrics
- **UAR** (Unweighted Average Recall = macro recall) — field-standard SER metric, treats all classes equally
- **Weighted F1** — accounts for class imbalance
- Accuracy is a secondary sanity check only

### Reproducibility
All code must set `RANDOM_SEED = 42`:

```python
import random, numpy as np
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

## Sprint Roadmap

| Sprint | Focus |
|--------|-------|
| Sprint 1 | Data cleaning, EDA, classical baseline (pre-extracted features) |
| Sprint 2 | Feature engineering: TF-IDF + MFCC/eGeMAPS audio + early fusion |
| Sprint 3 | Deep learning: Wav2Vec2/HuBERT, CRNN, DistilBERT, multimodal fusion |
| Sprint 4 | Ablation, error analysis, ensembling, report polish |

## Branch Workflow

- `main` is always stable — never commit directly
- Feature branches: `feature/<short-description>`
- Open PR against `main`, require at least one teammate review before merge
- Commit messages: imperative present tense (`Add confusion matrix plotter`)
