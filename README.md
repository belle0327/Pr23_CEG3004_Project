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
To transform variable-length temporal frames into a fixed-length **645-dimensional** vector, five statistics are computed across the time axis (`axis=1`) for every individual feature row:

* **Mean & Standard Deviation**: Capture the average spectral shape and variability of the signal.
* **Median**: Provides a central tendency robust to transient noise or outliers.
* **25th & 75th Percentiles**: Describe the dynamic range and the spread of the feature distribution.


---

### 4. Model
The final model is a **Support Vector Machine (SVM)**, implemented via the `SVC` (Support Vector Classifier) class.
```
Pipeline:
  └─ StandardScaler()
  └─ SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')
```
- **StandardScaler** — normalises all features to zero mean and unit variance before the SVM
- **SVC with RBF kernel** — captures non-linear decision boundaries in the high-dimensional feature space
- **`class_weight='balanced'`** — automatically adjusts class weights inversely proportional to class frequency. This was a deliberate design choice to handle any class imbalance in the training set (a form of experimental discipline), preventing the model from biasing predictions toward more common classes.

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

> These results are produced by running `notebooks/group23_ceg3004_project_colab.py` **as submitted** — the final version of the code with all TODO sections completed.
 
| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | ~0.60 (0.592) |
| **Macro-F1 Score** | ~0.57 (0.572) |
 
Evaluated on a stratified 80/20 train-validation split (`random_state=42`). This reflects the best configuration identified through the experiments in [`experiment.md`](./experiment.md) — earlier configurations scored as low as ~0.44 accuracy with no preprocessing and the baseline Logistic Regression model.

---

## Output Files

- `Pr_23_model.joblib`
- `Pr_23_predictions.csv`

---

## Reproducibility
