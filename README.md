# Speech Emotion Recognition — IEMOCAP

> **Introduction to Data Science · Bar-Ilan University**
> 8-Week Program Increment · 4 Sprints · Team of 3

Multimodal Speech Emotion Recognition (SER) system that classifies utterances into **4 emotion classes** (Angry, Happy, Neutral, Sad) using the [IEMOCAP](https://huggingface.co/datasets/AbstractTTS/IEMOCAP) corpus.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Setup](#setup)
3. [Label Mapping](#label-mapping)
4. [Folder Layout](#folder-layout)
5. [Branch & PR Workflow](#branch--pr-workflow)
6. [Sprint Roadmap](#sprint-roadmap)

---

## Dataset

**Source:** HuggingFace [`AbstractTTS/IEMOCAP`](https://huggingface.co/datasets/AbstractTTS/IEMOCAP) — ~1.4 GB, downloaded on first use and cached locally.

| Statistic | Value |
|---|---|
| Total utterances | 10,039 |
| Sessions | 5 (each with a unique speaker pair) |
| Male / Female speakers | 5,239 / 4,800 |
| Scripted / Improvised | 5,255 / 4,784 |

The dataset includes embedded audio, transcriptions, per-emotion soft-label probabilities, continuous VAD scores (EmoVal, EmoAct, EmoDom), and pre-extracted acoustic features (`pitch_mean`, `pitch_std`, `rms`, `relative_db`, `speaking_rate`).

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset (first run only):

```python
from datasets import load_dataset
ds = load_dataset("AbstractTTS/IEMOCAP", split="train")
```

---

## Label Mapping

The standard 4-class IEMOCAP benchmark collapses 10 original labels so that results are directly comparable across published work.

| Original label | 4-class label | Reason |
|---|---|---|
| `ang` (angry) | **angry** | Direct mapping |
| `hap` (happy) | **happy** | Direct mapping |
| `exc` (excited) | **happy** | Heavily overlaps with happy in acoustic features and annotator soft labels — merging reduces label noise without losing signal |
| `neu` (neutral) | **neutral** | Direct mapping |
| `sad` (sad) | **sad** | Direct mapping |
| `fru` (frustrated) | **dropped** | Largest dropped class (2,917 rows, ~29%). Sits between angry and neutral in VAD space with no clean boundary |
| `xxx` / `other` | **dropped** | No annotator agreement |
| `dis`, `fea`, `sur` | **dropped** | Too few samples (disgust = 2, fear = 107, surprise = 110) |

**After mapping: 6,877 utterances retained** (31.5% dropped).

---

## Folder Layout

```
.
├── data/                   # gitignored — cached dataset files
├── models/                 # gitignored — saved model weights
├── notebooks/              # Jupyter notebooks
│   ├── 01_baseline.ipynb
│   └── 02_eda.ipynb
├── reports/
│   └── figures/            # Saved plots
├── src/                    # Source modules
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Branch & PR Workflow

- `main` is always stable — **never commit directly**
- Feature branches: `feature/<short-description>`
- Open a PR against `main`; at least one teammate reviews before merge
- Commit messages: imperative present tense (`Add confusion matrix plotter`)

```bash
git checkout main && git pull
git checkout -b feature/my-task
# ... implement ...
git add <specific-files>
git commit -m "Add my feature"
git push -u origin feature/my-task
```

---

## Sprint Roadmap

| Sprint | Focus |
|---|---|
| **Sprint 1** | Data cleaning, EDA, classical baseline |
| **Sprint 2** | Feature engineering: TF-IDF + MFCC/eGeMAPS + early fusion |
| **Sprint 3** | Deep learning: Wav2Vec2/HuBERT, CRNN, DistilBERT, multimodal fusion |
| **Sprint 4** | Ablation, error analysis, ensembling, report polish |

---

## References

Busso, C. et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation, 42*(4), 335–359.
