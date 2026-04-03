# Pr23_CEG3004_Project3

# CEG3004 DSP Mini-Project: Environmental Sound Classification

## Project Overview
This project implements a robust environmental sound classification pipeline using DSP feature engineering and machine learning.

The goal is to classify audio clips into 50 environmental sound classes and maintain performance under:
- clean audio
- noisy audio
- band-limited audio

---

## Objectives
Based on the project briefing, the objectives are:
- Train on labeled environmental sound data
- Extract meaningful DSP features
- Build a machine learning classifier
- Improve robustness under distortions

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
- Remove NaN values
- Trim silence
- Pad/truncate to 5 seconds
- Apply pre-emphasis
- Peak normalization

### 2. Feature Extraction
Features used:

**Cepstral Features**
- MFCC (20)
- Delta
- Delta-Delta

**Spectral Features**
- Log-Mel Spectrogram (64)
- Spectral centroid
- Bandwidth
- Rolloff

**Temporal Features**
- Zero Crossing Rate
- RMS Energy

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
