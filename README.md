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

The final model is a **soft‑voting ensemble** that combines three different classifiers:

```
Pipeline:
  └─ StandardScaler()
  └─ VotingClassifier (Voting='Soft')
      ├─ Logistic Regression (C=1, balanced)
      ├─ SVM (RBF Kernel, C=10, balanced)
      └─ Random Forest (n=200)
```
---


**Why soft voting?**  
Each classifier outputs class probabilities; the ensemble averages them and picks the class with the highest mean probability. This reduces individual model variance and captures complementary decision boundaries.

---

## Experiments

All experiments used an 80/20 stratified train‑validation split (`random_state=42`) and the **final feature extraction pipeline** (F5: 645‑dim features with 5‑statistic pooling).

| Model | Validation Accuracy | Macro‑F1 |
|-------|---------------------|-----------|
| Logistic Regression (baseline) | 0.60 | ~0.59 |
| Random Forest (n=200) | 0.60 | ~0.58 |
| SVM (RBF, C=1) | 0.50 | ~0.46 |
| SVM (RBF, C=10, balanced) | 0.59 | ~0.57 |
| SVM (RBF, C=20) | 0.59 | ~0.57 |
| SVM (RBF, C=30) | 0.59 | ~0.57 |
| Gradient Boosting (n=200) | (too slow, no result) | — |
| **Voting Ensemble (soft)** | **0.62** | **~0.63** |

### Final Decision

The **Voting Ensemble** was selected as the final model because it achieved the highest validation accuracy (0.62) and Macro‑F1 (~0.63) across all tested configurations.  
- Single SVMs plateaued at 0.59 accuracy regardless of C.  
- Random Forest and Logistic Regression each scored 0.60 but made different error patterns.  
- Soft voting averages their probability outputs, yielding a robust improvement.

> **Note:** Data augmentation was experimented with but **removed** from the final pipeline – adding augmented copies to the already small training set (40 clips/class) caused overfitting and did not improve validation performance.

---

## Results

> These results are produced by running `notebooks/group23_ceg3004_project_colab.py` **as submitted** – the final version of the code with all TODO sections completed.

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **0.62** |
| **Macro‑F1 Score** | **~0.63** |

Evaluated on a stratified 80/20 train‑validation split (`random_state=42`). This reflects the best configuration identified through the experiments in [`experiment.md`](./experiment.md) – earlier configurations scored as low as ~0.44 accuracy with no preprocessing and the baseline Logistic Regression model.

---

## Output Files

- `Pr_23_model.joblib`  (the trained Voting Ensemble pipeline)
- `Pr_23_predictions.csv`

---

## Results

> These results are produced by running `notebooks/group23_ceg3004_project_colab.py` **as submitted** — the final version of the code with all TODO sections completed.
 
| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | ~0.63  |
| **Macro-F1 Score** | ~0.62  |
 
Evaluated on a stratified 80/20 train-validation split (`random_state=42`). This reflects the best configuration identified through the experiments in [`experiment.md`](./experiment.md) — earlier configurations scored as low as ~0.44 accuracy with no preprocessing and the baseline Logistic Regression model.

---

## Output Files

- `Pr_23_model.joblib`
- `Pr_23_predictions.csv`

---

## Reproducibility
Everything runs on **Google Colab** — no local installation or manual dataset setup is required. The notebook handles dependency installation and dataset download automatically.
 
### Step 1: Open the Notebook in Google Colab
 
Click the button below, or go to [colab.research.google.com](https://colab.research.google.com), choose **File → Upload notebook**, and upload `notebooks/group23_ceg3004_project_colab.py` from this repository.
 
### Step 2: Run All Cells in Order (Top to Bottom)
 
The notebook is fully self-contained. Running it sequentially will:
 
1. Install all required libraries (`librosa`, `scikit-learn`, `gdown`, etc.)
2. **Automatically download and extract the dataset** from Google Drive — no manual upload needed
3. Verify the dataset structure
4. Extract features for all 2,000 training clips
5. Train the SVM pipeline and print the validation report
6. Save and auto-download `Pr_23_model.joblib` to your computer
7. Generate predictions on the submission set and auto-download `Pr_23_predictions.csv`
 
> **If the automatic dataset download fails** (e.g., due to Google Drive quota limits), you can manually download it from [this link](https://drive.google.com/file/d/1bceZrbOMPSXTTTMBx8XqDBwsSMussPHj/view?usp=sharing) and upload it to Colab using the file panel on the left sidebar.
 
### Step 3: Collect Your Output Files
 
Upon completion, two files are automatically downloaded to your computer:
- `Pr_23_model.joblib`
- `Pr_23_predictions.csv`
 
> **Note:** Do **not** modify `clip_id` values in the predictions CSV. Only modify code in sections marked `🟨 STUDENT TODO`.
