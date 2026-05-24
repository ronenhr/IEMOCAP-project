# Speech Emotion Recognition — IEMOCAP

> **Introduction to Data Science · Engineering Faculty, Bar-Ilan University**
> 8-Week Program Increment · 4 Sprints · Team of 3

We build a multimodal Speech Emotion Recognition (SER) system that classifies
utterances into **4 emotion classes** (Angry, Happy, Neutral, Sad) using the
[IEMOCAP](https://huggingface.co/datasets/AbstractTTS/IEMOCAP) corpus.
Results are evaluated with a **speaker-independent** Leave-One-Session-Out
protocol to prevent data leakage — a deliberate design choice that makes our
numbers directly comparable to published baselines.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Setup](#setup)
3. [Getting the Data](#getting-the-data)
4. [Labelling Decisions](#labelling-decisions)
5. [Evaluation Protocol](#evaluation-protocol)
6. [Folder Layout](#folder-layout)
7. [Branch & PR Workflow](#branch--pr-workflow)
8. [Sprint Roadmap](#sprint-roadmap)
9. [Reproducibility](#reproducibility)
10. [References](#references)

---

## Dataset

We use two complementary sources that cover the same 10,039 utterances:

| Source | What it contains | Size |
|--------|-----------------|------|
| **Local CSV** (`iemocap_full_dataset.csv`) | Session, method (scripted/improv), gender, emotion label, annotator count & agreement, file path | ~1 MB — already in repo |
| **HuggingFace parquet** ([AbstractTTS/IEMOCAP](https://huggingface.co/datasets/AbstractTTS/IEMOCAP)) | Embedded audio, transcription, per-emotion soft-label probabilities, continuous EmoVal/EmoAct/EmoDom, pre-extracted features: `pitch_mean`, `pitch_std`, `rms`, `relative_db`, `speaking_rate` | ~1.4 GB — downloaded separately |

### Key dataset facts

| Statistic | Value |
|-----------|-------|
| Total utterances | 10,039 |
| Scripted / Spontaneous | 5,255 / 4,784 |
| Male / Female speakers | 5,098 / 4,941 |
| No-agreement rows (`xxx`) | 2,507 (~25%) — **dropped** |
| Sessions | 5 (each with a unique speaker pair) |

---

## Setup

### Local (Anaconda — recommended)

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. Create and activate a dedicated conda environment
conda create -n ser python=3.10
conda activate ser

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Getting the Data

The raw audio and HuggingFace parquet are **not** committed to this repo.
Each team member downloads them once — the `datasets` library handles caching.

```python
from datasets import load_dataset

# First run downloads ~1.4 GB and caches it locally.
# Every subsequent call reads from cache instantly.
ds = load_dataset("AbstractTTS/IEMOCAP", split="train")
```

The local metadata CSV (`iemocap_full_dataset.csv`) is already tracked in
the repo and requires no download.

---

## Labelling Decisions

### 4-class mapping (standard IEMOCAP benchmark)

The IEMOCAP research community collapses to four classes so that results are
directly comparable across papers. We follow this convention.

| Original label | 4-class label | Reason |
|---------------|---------------|--------|
| `ang` (angry) | **angry** | Direct mapping |
| `hap` (happy) | **happy** | Direct mapping |
| `exc` (excited) | **happy** | Excited and happy overlap heavily in the original confusion matrix (Busso et al., 2008) |
| `neu` (neutral) | **neutral** | Direct mapping |
| `sad` (sad) | **sad** | Direct mapping |
| `xxx` (no agreement) | **dropped** | ~25% of rows; training on noise hurts generalisation |
| `fru`, `dis`, `fea`, `sur` | **dropped** | Classes are too small to learn reliably (e.g., disgust = 2 samples) |

After cleaning, the usable 4-class set is approximately:
**Neutral 1,708 · Angry · Happy+Excited 1,636 · Sad 1,084**
(exact counts printed by the data loader at runtime).

---

## Evaluation Protocol

### Split — Leave-One-Session-Out (LOSO)

IEMOCAP has 5 sessions, each recorded by a **unique pair of speakers**.
We hold out one full session as the test set and train on the remaining four,
then rotate. This guarantees that **no speaker ever appears in both train and
test** — a requirement for honest, generalisation-testing results.

> Randomly splitting rows is incorrect for IEMOCAP: the same speaker would
> appear on both sides of the split, inflating scores artificially.

### Primary metrics

| Metric | Why we use it |
|--------|--------------|
| **Weighted F1** | Accounts for class imbalance; reflects overall system performance |
| **UAR** (Unweighted Average Recall = macro recall) | Treats every class equally regardless of frequency; the field-standard SER metric |

Accuracy is computed as a secondary sanity check only.

---

## Folder Layout

```
.
├── data/                       # gitignored — place downloaded dataset files here
│   └── .gitkeep
├── models/                     # gitignored — saved checkpoints and weights
│   └── .gitkeep
├── notebooks/                  # Jupyter notebooks for EDA and experiments
├── reports/
│   └── figures/                # Confusion matrices and other saved plots
├── src/                        # Source code modules (added during Sprint 1)
├── iemocap_full_dataset.csv    # Local metadata CSV (tracked in git)
├── .gitignore
├── README.md
└── requirements.txt
```

`data/` and `models/` are gitignored — never commit dataset files or model weights.

---

## Branch & PR Workflow

| Rule | Detail |
|------|--------|
| `main` | Always stable and runnable. **Never commit directly.** |
| Feature branches | One branch per task: `feature/<short-description>` |
| PRs | Open a PR against `main`. **At least one teammate reviews before merge.** |
| Commit messages | Imperative present tense: `Add confusion matrix plotter` |
| No secrets or data | Never commit `.env`, dataset files, or model weights |

```bash
# Start a new task
git checkout main && git pull
git checkout -b feature/my-task

# ... implement, test ...

git add <specific-files>
git commit -m "Add my feature"
git push -u origin feature/my-task
# → open PR on GitHub → request review → merge after approval
```

**Everyone commits every week.** The course graders read commit history to
assess individual contribution — ensure no team member goes a week without a
meaningful, well-described commit.

---

## Sprint Roadmap

| Phase | Primary Focus | Expected Outcome |
|-------|--------------|-----------------|
| **PI Planning** | Align on mission, metric, and split. Set up repo and board. | Shared GitHub Projects backlog; repo skeleton; leakage-free split agreed. |
| **Sprint 1** | Data cleaning, EDA, classical baseline (pre-extracted features). | Clean 4-class dataset; EDA figures; first honest UAR/F1 number; report Intro + Methods drafted. |
| **Sprint 2** | Feature engineering: TF-IDF text + MFCC/eGeMAPS audio + early fusion. | Multiple models compared in one table; fusion beats single-modality; **mid presentation**. |
| **Sprint 3** | Deep learning: Wav2Vec2/HuBERT embeddings, CRNN, DistilBERT text, multimodal fusion. | Best model locked; clear improvement over baseline; Experiments section growing. |
| **Sprint 4** | Ablation analysis, error analysis, ensembling, report and deck polish. | Final report; **final presentation**; clean tagged release. |

### Definition of Done (per sprint)

- Code reviewed by the sprint's Tech Lead and merged via Pull Request.
- Runs end-to-end from the README on a teammate's machine / fresh Colab.
- Results logged with the shared metrics module (Weighted-F1 + UAR + confusion matrix).
- The relevant report section updated in the same sprint.

---

## Reproducibility

All code uses **`RANDOM_SEED = 42`**. Set the same seed in every notebook:

```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

---

## References

- Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., & Narayanan, S. S. (2008).
  **IEMOCAP: Interactive emotional dyadic motion capture database.**
  *Language Resources and Evaluation, 42*(4), 335–359.
