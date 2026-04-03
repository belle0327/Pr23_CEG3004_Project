# Pr23_CEG3004_Project

# CEG3004 DSP Mini-Project: Environmental Sound Classification

## Project Overview
This project implements a robust audio classification pipeline for **Environmental Sound Classification (ESC-50)**. The goal is to classify audio clips into **50 environmental sound classes** while maintaining strong performance under three distinct conditions:
 
- **Clean** — original, unmodified audio
- **Noisy** — additive noise applied to the signal
- **Band-limited** — frequency content restricted to a narrower range
 
The final performance score is weighted: **50% Clean + 25% Noisy + 25% Band-limited**.

---

## Objectives
Based on the project briefing, the objectives are:
- Train on labeled environmental sound data
- Extract meaningful DSP features
- Build a machine learning classifier
- Improve robustness under distortions

---
## Environment
 
This project runs entirely on **Google Colab** — no local setup needed. All dependencies are installed automatically by the first cell of the notebook.
 
| Library | Purpose |
|---|---|
| `librosa` | Audio loading, feature extraction (MFCC, mel spectrogram, spectral features) |
| `scikit-learn` | ML pipeline, SVM classifier, StandardScaler, metrics |
| `numpy` | Numerical operations and feature pooling |
| `pandas` | Loading and managing CSV metadata |
| `soundfile` | WAV file I/O |
| `joblib` | Saving and loading the trained model |
| `tqdm` | Progress bars during feature extraction |
 
---
## Dataset
- 2000 audio clips
- 50 classes
- 5 seconds per clip
- Mono audio

Submission dataset includes:
- clean
- noisy
- band-limited versions

---

## Pipeline Design

### 1. Preprocessing
Applied to every audio clip before feature extraction:
 
1. **NaN removal** — `np.nan_to_num()` ensures no invalid values
2. **Silence trimming** — `librosa.effects.trim(top_db=25)` removes leading/trailing silence
3. **Fixed-length padding/truncation** — all clips standardised to exactly **5 seconds** at 16 kHz
4. **Pre-emphasis filter** — `librosa.effects.preemphasis()` boosts high-frequency content to counteract spectral tilt, improving MFCC quality
5. **Peak normalisation** — divides by peak amplitude so all clips are on the same scale

### 2. Feature Extraction
All features are extracted from 16 kHz mono audio using `n_fft=1024` and `hop_length=256`. 

| Feature Group | Feature Component | Base Dim ($N$) | Stat Multiplier | Final Dim |
| :--- | :--- | :---: | :---: | :---: |
| **Cepstral** | MFCC (Row-Normalized) | 20 | $\times 5$ | 100 |
| | Delta MFCC (d1) | 20 | $\times 5$ | 100 |
| | Delta-Delta MFCC (d2) | 20 | $\times 5$ | 100 |
| **Spectral** | Log-Mel Spectrogram (Row-Normalized) | 64 | $\times 5$ | 320 |
| | Spectral Centroid | 1 | $\times 5$ | 5 |
| | Spectral Bandwidth | 1 | $\times 5$ | 5 |
| | Spectral Rolloff | 1 | $\times 5$ | 5 |
| **Temporal** | Zero Crossing Rate | 1 | $\times 5$ | 5 |
| | RMS Energy | 1 | $\times 5$ | 5 |
| **Total** | **Full Feature Vector** | **129** | — | **645** |

---

### 3. Feature Pooling
For each feature:
- Mean
- Standard deviation
- Median
- 25th percentile
- 75th percentile

---

### 4. Model

Pipeline:
- StandardScaler
- SVM (RBF Kernel)

Final model: SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')


---

## Experiments

Tested:
- SVM (C = 10, 20, 30)
- Random Forest
- Feature selection (SelectKBest)
- Augmentation (NOT used in final)

### Final Decision
- C=10 chosen (same performance, simpler)
- Random Forest worse
- Feature selection no improvement
- Augmentation reduced performance → removed

---

## Results

Validation:
- Accuracy ≈ 0.60
- Macro-F1 ≈ 0.57

---

## Output Files

- `Pr_23_model.joblib`
- `Pr_23_predictions.csv`

---

## Reproducibility
