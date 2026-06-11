# Label Decision Plots — Design Spec

**Date:** 2026-06-11  
**Notebook:** `notebooks/02_eda.ipynb`  
**Placement:** After existing soft-label heatmap (cell 36), before Step 6 Audio Analysis

## Goal

Add four plots to visually justify every label decision made in Step 5:
- Why `excited → happy` was merged
- Why `frustrated` was kept as its own class
- Why `xxx` was dropped (low annotator agreement)
- Why disgust, fear, surprise, other were dropped (too few samples)

## Data Source

All plots use the HuggingFace pre-extracted columns already present in `df_raw`:
- Acoustic: `pitch_mean`, `pitch_std`, `rms`, `relative_db`, `speaking_rate`
- Dimensional: `EmoAct`, `EmoVal`, `EmoDom`
- Label quality: `agreement`, `n_annotators`
- Emotion: `major_emotion` (all 10 original classes)

No new data loading or librosa computation is needed.

## Color Scheme

Consistent with existing notebook palette (cells 29–31):

| Decision | Color | Classes |
|---|---|---|
| Kept | `#4CAF50` green | angry, frustrated, neutral, sad, happy |
| Merged → happy | `#FF9800` orange | excited |
| Dropped (xxx) | `#e53935` red | xxx |
| Dropped (too few) | `#9E9E9E` grey | disgust, fear, surprise, other |

## Plot 1 — Overview: All 10 Classes × 3 Acoustic Features

**Type:** Horizontal strip plot + box overlay, 1 row × 3 columns  
**Features:** `pitch_mean`, `rms`, `speaking_rate`  
**Classes:** All 10 original `major_emotion` values on y-axis  
**Coloring:** By decision (4 colors above)  
**Purpose:** Full emotion space at a glance — readers immediately see excited/happy overlap, frustrated's intermediate position, and xxx/minority classes' spread  
**Sample cap:** 200 points per class (random sample) for readability — `xxx` and `frustrated` have 2500+ rows and would overwhelm the strip otherwise

## Plot 2 — Merge Justification: excited vs happy

**Type:** Violin plots, 2 rows × 3 columns  
**Features:** `pitch_mean`, `rms`, `speaking_rate`, `pitch_std`, `relative_db`, `EmoVal`  
**Classes:** `excited` (orange), `happy` (green) only  
**Purpose:** Violin shapes nearly stack → classifier cannot distinguish these classes → merge is acoustically justified  
**Markdown note:** Explains that overlapping distributions make these classes indistinguishable at the feature level

## Plot 3 — Keep Frustrated: angry / frustrated / neutral

**Type:** Violin plots, 2 rows × 3 columns (same layout as Plot 2)  
**Features:** `pitch_mean`, `rms`, `speaking_rate`, `pitch_std`, `relative_db`, `EmoAct`  
**Classes:** `angry` (`#d62728` red), `frustrated` (`#ff7f0e` orange), `neutral` (`#1f77b4` blue) — uses Step 6 AUDIO_PALETTE so the three classes are visually distinct despite all being "kept"  
**Purpose:** Frustrated sits visibly between angry and neutral on pitch and RMS — neither close enough to merge into either, distinct enough to warrant its own class  
**Markdown note:** Explains that merging into angry would introduce high-arousal/negative-valence confusion; merging into neutral would discard a meaningful class

## Plot 4 — Drop Justification: Two Panels

**Type:** Two-panel figure  

**Left panel — Agreement score boxplot:**  
- Horizontal boxplot of `agreement` for all 10 original classes, ordered by median  
- `xxx` highlighted in red, all others in their decision color  
- Vertical dashed line at the minimum acceptable agreement level  
- Shows `xxx` is a clear low-agreement outlier regardless of acoustic profile  

**Right panel — Sample count bar chart:**  
- Horizontal bar chart, all 10 classes, ordered by count  
- Colored by decision  
- Horizontal dashed line at 200 samples (minimum threshold for reliable learning)  
- Disgust (2), other (3), fear (40), surprise (107) all fall below it  
- `xxx` shown with a separate annotation noting it was dropped for agreement reasons, not count

## Insertion Point in Notebook

Insert after cell 36 (the markdown cell following the soft-label heatmap). The new block is:

```
[markdown] --- heading: "Acoustic Feature Evidence"
[code]     Plot 1 — overview strip+box
[markdown] observation text for Plot 1
[code]     Plot 2 — excited vs happy violins
[markdown] observation text for Plot 2
[code]     Plot 3 — angry / frustrated / neutral violins
[markdown] observation text for Plot 3
[code]     Plot 4 — drop justification two panels
[markdown] observation text for Plot 4
```

## Out of Scope

- No librosa computation — only pre-extracted HuggingFace columns
- No changes to Step 6 audio analysis section
- No changes to Step 5 label mapping code
- No new notebook files
