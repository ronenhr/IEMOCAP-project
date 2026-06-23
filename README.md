# IEMOCAP Speech Emotion Recognition

> **Introduction to Data Science · Bar-Ilan University**  
> Multimodal Speech Emotion Recognition project using audio, text, and late fusion.

**Team:** Ben Shakarof, Ronen Hristoforov, Evyatar Azoulay

---

## Overview

This project builds a Speech Emotion Recognition (SER) system on the IEMOCAP corpus.  
The final system compares three approaches:

1. **Audio-only model:** fine-tuned HuBERT on raw speech audio.
2. **Text-only model:** fine-tuned DistilBERT on Whisper transcripts.
3. **Multimodal late fusion:** weighted averaging of HuBERT and DistilBERT prediction probabilities.

The main research question is whether combining acoustic and textual information improves emotion classification compared with either modality alone.

---

## Final Emotion Classes

The project uses a **5-class** emotion setup:

- angry
- happy
- neutral
- sad
- frustrated

The original `excited` label is merged into `happy`, while minority labels are dropped.

| Original label | Final label | Decision |
|---|---|---|
| `angry` | angry | kept |
| `happy` | happy | kept |
| `excited` | happy | merged with happy |
| `neutral` | neutral | kept |
| `sad` | sad | kept |
| `frustrated` | frustrated | kept as a fifth class |
| `disgust`, `fear`, `surprise`, `other` | - | dropped due to very small sample sizes |

After mapping, the final dataset contains **9,794 utterances**.

---

## Evaluation Protocol

All final models use the same **session-level speaker-independent holdout split**:

| Split | Sessions | Purpose |
|---|---|---|
| Train | Ses01, Ses02, Ses03 | model training |
| Validation | Ses04 | hyperparameter selection and fusion weight search |
| Test | Ses05 | final held-out evaluation |

This split prevents speaker leakage: speakers from the test session are never seen during training.

The primary metric is **UAR**: Unweighted Average Recall, also known as balanced accuracy.  
UAR is used because the emotion classes are imbalanced and each class should contribute equally to the final score.

---

## Final Results

All results below are measured on the same held-out test session, **Ses05**.

| Model | Modality | Test Accuracy | Test UAR | Test Macro F1 |
|---|---|---:|---:|---:|
| HuBERT Audio | Audio | 0.4934 | 0.5174 | 0.4955 |
| DistilBERT Text | Text | 0.5137 | 0.5408 | 0.4993 |
| HuBERT + DistilBERT Late Fusion | Audio + Text | **0.5628** | **0.5915** | **0.5535** |

The best fusion configuration was selected on the validation set:

| Audio weight | Text weight | Validation UAR | Test UAR |
|---:|---:|---:|---:|
| 0.45 | 0.55 | 0.6082 | 0.5915 |

Fusion improvements:

| Comparison | UAR improvement | Macro F1 improvement |
|---|---:|---:|
| Fusion vs HuBERT Audio | +0.0741 | +0.0579 |
| Fusion vs DistilBERT Text | +0.0507 | +0.0541 |

These results support the conclusion that audio and text provide complementary information for emotion recognition.

---

## Repository Structure

```text
IEMOCAP-project/
├── Sprint_2/
├── data/
├── docs/
├── models/
├── notebooks/
│   ├── 01_download_dataset.ipynb
│   ├── 02_eda.ipynb
│   ├── 02a_transcribe_colab.ipynb
│   ├── 03_text_features_and_baseline.ipynb
│   ├── 03b_Audio_EDA_and_Analysis.ipynb
│   ├── 04_hubert_audio_LOSO.ipynb
│   ├── 04_text_bert.ipynb
│   └── 05_fusion_baseline.ipynb
├── outputs/
│   ├── fusion_bert_hubert/
│   ├── hubert_audio_baseline/
│   ├── whisper_text_features/
│   └── shared_split_indices.csv
├── README.md
└── requirements.txt
```

Note: `04_hubert_audio_LOSO.ipynb` is a historical filename. The final audio notebook uses a **session-level holdout split**, not full LOSO cross-validation.

---

## Main Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/02_eda.ipynb` | Exploratory data analysis, label mapping, class balance, VAD analysis, acoustic feature analysis |
| `notebooks/04_hubert_audio_LOSO.ipynb` | Final HuBERT audio classifier using session-level holdout |
| `notebooks/04_text_bert.ipynb` | Final DistilBERT text classifier using Whisper transcripts |
| `notebooks/05_fusion_baseline.ipynb` | Late fusion of HuBERT and DistilBERT probabilities |

Additional notebooks:

| Notebook | Purpose |
|---|---|
| `notebooks/01_download_dataset.ipynb` | Dataset download and setup |
| `notebooks/02a_transcribe_colab.ipynb` | Whisper transcription pipeline |
| `notebooks/03_text_features_and_baseline.ipynb` | Earlier text-feature and baseline analysis |
| `notebooks/03b_Audio_EDA_and_Analysis.ipynb` | Additional audio EDA |

---

## Methodology

### 1. Exploratory Data Analysis

The EDA notebook examines:

- raw emotion label distribution
- 5-class label mapping
- class imbalance
- session distribution
- gender and scripted/improvised distributions
- VAD scores: valence, activation, dominance
- acoustic patterns such as pitch, RMS, MFCCs, spectral centroid, and ZCR
- transcript statistics

Key EDA conclusions:

- `frustrated` is too large to drop and is retained as a fifth class.
- `excited` strongly overlaps with `happy` and is merged into it.
- all five final classes appear in every session.
- class imbalance motivates the use of UAR as the primary metric.

### 2. HuBERT Audio Model

The audio model fine-tunes:

```text
facebook/hubert-base-ls960
```

Main components:

- raw waveform input
- HuBERT feature extractor
- mean and max pooling over hidden states
- weighted loss for class imbalance
- audio augmentation during training
- validation UAR checkpointing
- export of probability CSVs for fusion

Final HuBERT test performance:

```text
Test UAR      = 0.5174
Test Accuracy = 0.4934
Test Macro F1 = 0.4955
```

### 3. DistilBERT Text Model

The text model fine-tunes:

```text
distilbert-base-uncased
```

Main components:

- Whisper transcript input
- DistilBERT sequence classification head
- weighted loss for class imbalance
- small hyperparameter search
- final model selected by validation UAR
- export of probability CSVs for fusion

Final DistilBERT test performance:

```text
Test UAR      = 0.5408
Test Accuracy = 0.5137
Test Macro F1 = 0.4993
```

### 4. Late Fusion

The fusion notebook combines the probability vectors from the HuBERT and DistilBERT models:

```text
fusion_probs = audio_weight * audio_probs + text_weight * text_probs
```

The best weight is selected by validation UAR.  
The final selected weights are:

```text
Audio weight = 0.45
Text weight  = 0.55
```

Final fusion test performance:

```text
Test UAR      = 0.5915
Test Accuracy = 0.5628
Test Macro F1 = 0.5535
```

---

## Outputs

Important output files include:

```text
outputs/hubert_audio_baseline/
├── hubert_val_predictions_with_probs.csv
├── hubert_test_predictions_with_probs.csv
├── hubert_final_metrics.json
├── bert_text_val_predictions_with_probs.csv
├── bert_text_test_predictions_with_probs.csv
└── bert_text_final_metrics.json

outputs/fusion_bert_hubert/
├── fusion_validation_results.csv
├── fusion_test_predictions.csv
├── fusion_final_metrics.json
├── fusion_weight_search.png
└── fusion_confusion_matrix.png
```

The fusion notebook expects HuBERT and DistilBERT prediction files to be available before running.

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the deep learning notebooks, a GPU runtime is recommended. The audio and text notebooks were developed primarily for Google Colab / GPU execution.

---

## How to Run

Recommended order:

1. Run EDA:

   ```text
   notebooks/02_eda.ipynb
   ```

2. Run or inspect the HuBERT audio model:

   ```text
   notebooks/04_hubert_audio_LOSO.ipynb
   ```

3. Run or inspect the DistilBERT text model:

   ```text
   notebooks/04_text_bert.ipynb
   ```

4. Run fusion after both audio and text prediction CSVs exist:

   ```text
   notebooks/05_fusion_baseline.ipynb
   ```

The fusion notebook loads precomputed probability CSVs and does not retrain the audio or text models.

---

## Reproducibility Notes

- Random seed: `42`
- Final split: train Ses01-Ses03, validation Ses04, test Ses05
- Primary metric: UAR / balanced accuracy
- The test set is used only for final evaluation
- Fusion weights are selected using validation UAR only

---

## References

Busso, C. et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation, 42*(4), 335-359.
