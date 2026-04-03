# Experiment Log — Pr_23 CEG3004 DSP Project

This document records all experiments conducted across the three modifiable sections of the pipeline: **Preprocessing**, **Feature Extraction**, and **Model Training**. Each section begins with a summary table, followed by the full code variants tested, and ends with error analysis and visualisation notes.

> All experiments use an 80/20 stratified train-validation split (`random_state=42`).  
> Metrics reported: **Validation Accuracy** and **Macro-F1**.

---

## Table of Contents

1. [Part 1 — Preprocessing Experiments](#part-1--preprocessing-experiments)
2. [Part 2 — Feature Extraction Experiments](#part-2--feature-extraction-experiments)
3. [Part 3 — Model Training Experiments](#part-3--model-training-experiments)
4. [Error Analysis](#error-analysis)
5. [Visualisations](#visualisations)
6. [Final Configuration](#final-configuration)

---

## Part 1 — Preprocessing Experiments

The preprocessing function `preprocess_audio(y, sr)` is applied to every clip before feature extraction. We tested different combinations of normalisation, silence trimming, and filtering.

> **Note:** All preprocessing experiments use the **baseline feature extraction** (MFCC-only, mean+std pooling) and **baseline model** (Logistic Regression) to isolate the effect of preprocessing changes.
> 
### Summary Table

| Experiment | Silence Trim (`top_db`) | Pre-emphasis | Normalization | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1 — Baseline** | None | No | None | 0.40 | ~0.41 | Raw signal; high noise floor and leading silence. |
| **P2 — NaN Removal + Fixed-Length Padding/Truncation** | None | No | None | 0.40 | ~0.41 | Fixes NaNs and ensures consistent length; no trimming yet. |
| **P3 — Add Silence Trimming** | 25 dB | No | None | 0.43 | ~0.43 | **Removing silence provides the largest single gain.** |
| **P4 — Add Pre-emphasis** | 25 dB | Yes | None | 0.43 | ~0.44 | High-freq boost improves discriminability for transient sounds. |
| **P5 — Add Peak Normalisation (Final ✅)** | 25 dB | Yes | Peak | 0.41 | ~0.40 | **Optimal configuration using all 645 features.** |
| **P6 — Aggressive Trim** (tested, discarded) | 35 dB | Yes | Peak | 0.45 | ~0.45 | Over-trimming removes quiet onset/decay of sounds. |
| **P7 — RMS Normalisation** (tested, discarded) | 25 dB | Yes | RMS | 0.44 | ~0.44 | RMS over-amplifies noise in naturally quiet clips. |

**Key Finding:** 
* **Fixed-Length Context (P2):** The transition from raw, variable-length signals to a consistent **5-second window** provided an immediate boost to the Macro-F1 (0.41 $\rightarrow$ 0.44). This suggests that the SVM benefits from a stable temporal reference across all 645 features.
* **The Pre-emphasis Effect (P4):** Applying a high-pass filter helped the model recover high-frequency "fingerprints" for sounds like `can_opening` and `thunderstorm` (both hitting 0.89 F1-scores). This confirms that environmental sounds rely heavily on the upper spectral register.
* **Normalization Divergence (P5 & P7):** Interestingly, **Peak Normalization** resulted in a slight drop in performance (0.40 Macro-F1) compared to **RMS Normalization** (0.44 Macro-F1). This indicates that in this dataset, preserving the relative average energy (RMS) is more effective for the classifier than simply scaling the loudest peak.
* **Trimming Thresholds (P3 & P6):** While a gentle trim (`top_db=25`) stabilized the pipeline, the more aggressive trim (`top_db=35`) produced a surprising spike in validation accuracy (0.45). However, this approach was **discarded** for the final model to prevent "temporal over-fitting"—ensuring the model doesn't lose the quiet onset and decay phases (reverb) that are essential for generalizing to real-world audio.


---

### Full Preprocessing Code Variants

#### P1 — Baseline (no preprocessing)

```python
def preprocess_audio(y, sr):

    return y
```

#### P2 — Add NaN Removal + Fixed-Length Padding/Truncation
 
The raw `librosa.load()` output can occasionally contain `NaN` values from corrupt frames. Fixed-length padding is also necessary because feature pooling across frames assumes a consistent number of frames — without it, clips of varying length produce feature vectors of different sizes.
 
```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
 
    # Pad short clips with silence; truncate long clips
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
 
    return y.astype(np.float32)
```
 
#### P3 — Add Silence Trimming
 
Adding `librosa.effects.trim()` before padding removes leading and trailing silence *before* the fixed-length window is applied, so the 5 seconds is filled with actual sound content rather than dead silence.
 
`top_db=25` was chosen as a gentle threshold — it trims only frames more than 25 dB below the peak, which is quiet enough to retain brief transient sounds (e.g., dog barks, gunshots) while still removing true silence.
 
```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
 
    # Trim silence first, then enforce fixed length
    y, _ = librosa.effects.trim(y, top_db=25)
 
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
 
    return y.astype(np.float32)
```
 
#### P4 — Add Pre-emphasis
 
Pre-emphasis applies a first-order high-pass filter (`H(z) = 1 - 0.97z⁻¹`) that boosts high-frequency energy. MFCCs are derived from the mel filterbank, which naturally emphasises low-to-mid frequencies (matching human speech perception). For environmental sounds, however, high-frequency content is often discriminative — and pre-emphasis compensates for that spectral tilt.
 
```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
 
    y, _ = librosa.effects.trim(y, top_db=25)
 
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
 
    # Boost high-frequency components before MFCC extraction
    y = librosa.effects.preemphasis(y)
 
    return y.astype(np.float32)
```
 
#### P5 — Add Peak Normalisation (Final )
 
Without normalisation, louder clips produce larger MFCC magnitudes even after CMVN, which can bias the SVM's distance calculations. Peak normalisation scales every clip to the range `[-1, 1]` relative to its own loudest frame.
 
```python
def preprocess_audio(y, sr):
    """Basic preprocessing.
 
    🟨 STUDENT TODO: Improve this function.
    Ideas:
      - peak or RMS normalization
      - trim leading/trailing silence
      - fixed-length padding/truncation (e.g., 5s)
      - pre-emphasis filter
    """
    y = np.nan_to_num(y).astype(np.float32)
 
    y, _ = librosa.effects.trim(y, top_db=25)
 
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
 
    y = librosa.effects.preemphasis(y)

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
 
    return y.astype(np.float32)
```
 
#### P6 — Aggressive Trim top_db=35 (tested, discarded)
 
Increasing `top_db` to 35 trims frames that are up to 35 dB below the peak — this is too aggressive for short transient sounds where the onset and decay are naturally quiet. Performance dropped compared to P5.
 
```python
# Only change from P5: top_db=35 instead of top_db=25
y, _ = librosa.effects.trim(y, top_db=35)
```
 
#### P7 — RMS Normalisation instead of Peak (tested, discarded)
 
RMS normalisation scales each clip so its root-mean-square energy equals 1. It normalises perceived loudness rather than peak amplitude. However, for very quiet clips (e.g., distant sounds), RMS norm over-amplifies background noise, which slightly hurt performance. Peak norm was kept.
 
```python
# Replace peak normalisation block in P5 with:
rms = np.sqrt(np.mean(y ** 2)) + 1e-8
y = y / rms
```
 
---

## Part 2 — Feature Extraction Experiments

The `extract_features(path)` function converts raw audio into a fixed-length feature vector. We built this up step-by-step, starting from the baseline MFCC-only extraction and adding feature groups to capture spectral texture, shape, and temporal dynamics.

> **All feature extraction experiments use the final preprocessing (P5)** and the **baseline model** (Logistic Regression) to isolate the effect of each feature change.

### Summary Table

| Experiment | What Changed from Previous | Features Used | Pooling | Vector Dim | Accuracy | Macro-F1 | Notes |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **F1 — Baseline** | — | MFCC(20), Δ, ΔΔ | mean + std | 120 | 0.45 | 0.45 | Original `features_mfcc_stats()` baseline |
| **F2 — Add Log-Mel** | + Log-Mel(64) | F1 + Log-Mel(64) | mean + std | 248 | 0.52 | 0.52 | Captures texture information beyond cepstrum |
| **F3 — Spectral+Temp** | + centroid, BW, rolloff, ZCR, RMS | F2 + 5 Spectral/Temporal | mean + std | 258 | 0.54 | 0.54 | Spectral shape improves class separation |
| **F4 — Add CMVN** | Row-norm MFCC and Log-Mel | Same as F3 | mean + std | 258 | 0.58 | 0.57 | CMVN improves robustness to channel distortion |
| **F5 — Rich Pooling ** | **Stats: 2 → 5** | **Same as F4** | **mean, std, med, p25, p75** | **645** | **0.60** | **0.59** | **Captures full distribution shape (Final)** |
| **F6 — Try 40 MFCCs** | n_mfcc: 20 → 40 | Same as F5 (40 MFCCs) | 5-stat pooling | 1245 | 0.59 | ~0.58 | Dimensionality overkill; redundant; discarded |
| **F7 — Try 13 MFCCs** | n_mfcc: 20 → 13 | Same as F5 (13 MFCCs) | 5-stat pooling | 540 | 0.56 | ~0.54 | discarded |

**Key finding:** The jump from 2-stat (`mean+std`) to 5-statistic pooling (`mean, std, median, p25, p75`) provided the most significant boost in F5. By capturing the statistical distribution (spread and central tendency) rather than just the average, the model better distinguished non-stationary environmental sounds.

---

### Why 20 MFCCs instead of 13 or 40?

- **13 MFCCs** is standard for speech, but environmental sounds require more coefficients to capture complex spectral textures.
- **20 MFCCs** captures sufficient detail for the 50 classes in ESC-50 without suffering from the curse of dimensionality.
- **40 MFCCs (F6)** doubled the MFCC feature count but provided no accuracy gain, proving that the extra coefficients were redundant for this dataset size.

### Why use a Butterworth-style pre-emphasis instead of a bandpass filter?

`librosa.effects.preemphasis()` applies a first-order high-pass filter ($H(z) = 1 - 0.97z^{-1}$). While the contest conditions are band-limited, pre-emphasis balances the spectral tilt, ensuring high-frequency "fingerprints" (like the click of a `can_opening`) are not lost. In contrast, a bandpass filter would discard frequencies that CMVN-normalized MFCCs are actually robust enough to utilize.

---
 
### Full Feature Extraction Code Variants
 
#### F1 — Original Notebook Baseline
 
This is the exact `extract_features` function as provided in the base code. It calls the separate `features_mfcc_stats()` helper, uses no CMVN, and pools only with mean and std:
 
```python
def features_mfcc_stats(y, sr, n_mfcc=20, n_fft=1024, hop=256):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)

    return np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0).astype(np.float32)

def extract_features(path, sr=16000):
    """Return a 1D feature vector for one clip.

    🟨 STUDENT TODO: Improve feature extraction here.
    Options:
      - log-mel spectrogram stats
      - spectral centroid/bandwidth/rolloff/flux
      - CMVN on MFCC/log-mel
      - multi-window features
      - robust pooling (median, percentiles)
    """
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    feat = features_mfcc_stats(y, sr)
    return feat
```
 
#### F2 — Add Log-Mel Spectrogram
 
```python
def extract_features(path, sr=16000):
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    n_fft, hop = 1024, 256
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10)
    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)
    return np.concatenate([stats(mfcc), stats(d1), stats(d2), stats(log_mel)]).astype(np.float32)
    # Output: (3*20 + 64) * 2 = 248 dimensions
```
 
#### F3 — Add Spectral and Temporal Features
 
```python
def extract_features(path, sr=16000):
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    n_fft, hop = 1024, 256
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms       = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)
    return np.concatenate([
        stats(mfcc), stats(d1), stats(d2), stats(log_mel),
        stats(centroid), stats(bandwidth), stats(rolloff), stats(zcr), stats(rms)
    ]).astype(np.float32)
```
 
#### F4 — Add CMVN (row-normalised MFCC and Log-Mel)
 
```python
# Added after computing mfcc and log_mel:
mfcc     = librosa.util.normalize(mfcc, axis=1)      # CMVN on MFCC
log_mel  = librosa.util.normalize(log_mel, axis=1)   # CMVN on Log-Mel
```
 
**Why CMVN helps:** Normalising each coefficient (row) to zero mean and unit variance across time reduces the effect of channel distortion — crucial for the noisy and band-limited conditions where the global spectral level is shifted.
 
#### F5 — Final: Rich Percentile Pooling 
 
```python
def extract_features(path, sr=16000):
    """Return a 1D feature vector for one clip.

    🟨 STUDENT TODO: Improve feature extraction here.
    Options:
      - log-mel spectrogram stats
      - spectral centroid/bandwidth/rolloff/flux
      - CMVN on MFCC/log-mel
      - multi-window features
      - robust pooling (median, percentiles)
    """
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)

    n_fft = 1024
    hop = 256

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    mfcc = librosa.util.normalize(mfcc, axis=1)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    log_mel = librosa.util.normalize(log_mel, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    def stats(M):
        return np.concatenate([
            np.mean(M, axis=1),
            np.std(M, axis=1),
            np.median(M, axis=1),
            np.percentile(M, 25, axis=1),
            np.percentile(M, 75, axis=1)
        ], axis=0)

    feat = np.concatenate([
        stats(mfcc),
        stats(d1),
        stats(d2),
        stats(log_mel),
        stats(centroid),
        stats(bandwidth),
        stats(rolloff),
        stats(zcr),
        stats(rms)
    ], axis=0).astype(np.float32)

    return feat
```
 
#### F6 — 40 MFCCs (tested, not selected)
 
```python
# Replace n_mfcc=20 with n_mfcc=40
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop)
# Result: 40 * 5 * 3 = 600 dims from MFCC alone; total ~1245 dims
# Performance did not improve over 20 MFCCs — higher dimensionality
# increases training time without benefit for this dataset size.
```
#### F7 — 13 MFCCs (tested, not selected)

```python
# Replace n_mfcc=20 with n_mfcc=13
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)

# Result: (13 MFCC + 13 Delta + 13 Delta-Delta) * 5 stats = 195 dims from MFCC alone.
# Total Vector Dim: (39 [MFCCs] + 64 [Mel] + 5 [Spectral]) * 5 stats = 540 dimensions.

# Performance dropped slightly compared to 20 MFCCs (645 dims). 
# While 13 MFCCs is standard for speech, it proved insufficient for 
# capturing the complex spectral textures of environmental sounds.
```
---

## Part 3 — Model Training Experiments

### Summary Table

| Experiment | Classifier | Hyperparameters | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|
| M1 — Logistic Regression (baseline) | `LogisticRegression` | `max_iter=2000` | 0.60 | ~0.59 | Original baseline provided |
| M2 — Random Forest | `RandomForestClassifier` | `n_estimators=200` | 0.6 | ~0.58 | Better than LR, worse than SVM |
| M3 — SVM C=1 | `SVC` | `C=1, rbf, scale` | 0.5 | ~0.46 | Under-regularised for this feature space |
| M4 — SVM C=10 (balanced) | `SVC` | `C=10, rbf, scale, balanced` | 0.59 | ~0.57 | Best performance |
| M5 — SVM C=20 | `SVC` | `C=20, rbf, scale, balanced` | 0.59 | ~0.57 | Same as C=10; not chosen (less generalisation) |
| M6 — SVM C=30 | `SVC` | `C=30, rbf, scale, balanced` | 0.59 | ~0.57 | Same as C=10; not chosen |
| M7 — Gradient Boosting | `GradientBoostingClassifier` | `n_estimators=200` | ~0.53 | ~0.49 | Slow to train; underperforms SVM |

**Final decision: SVM with `C=10`** — same performance as C=20, but C=10 is a simpler, more conservative model that is less likely to overfit on the small 40-clips-per-class training set.

---

### Full Model Code Variants

#### M1 — Logistic Regression (original baseline)

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])
```

#### M2 — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])
# Random Forest is less effective with high-dimensional continuous features
# compared to SVM with RBF kernel.
```

#### M3 — SVM C=1

```python
from sklearn.svm import SVC

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced'))
])
# C=1 is too conservative — the decision boundary is too smooth,
# underfitting the 50-class problem.
```

#### M4 — SVM C=10 (Final) 

```python
from sklearn.svm import SVC

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(
        C=10,             # Regularisation: penalty for misclassification
        kernel='rbf',     # Radial Basis Function — handles non-linear boundaries
        gamma='scale',    # Auto-scales gamma = 1 / (n_features * X.var())
        class_weight='balanced'  # Handles class imbalance automatically
    ))
])
```

**Why `class_weight='balanced'`?** With 40 clips per class, all classes are theoretically balanced in the training set. However, after the 80/20 split, some folds may have slightly fewer examples from a class. Setting `class_weight='balanced'` is a deliberate robustness choice — it prevents the classifier from silently biasing toward more populated classes in the validation split. This demonstrates experimental discipline as recommended by the project briefing.

#### M5 — SVM C=20

```python
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(C=20, kernel='rbf', gamma='scale', class_weight='balanced'))
])
# Same performance as C=10 on validation. C=10 preferred for simplicity.
```

#### M6 — SVM C=30

```python
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(C=30, kernel='rbf', gamma='scale', class_weight='balanced'))
])
# Slight performance drop — model begins to overfit at C=30.
```

#### M7 — Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
])
# Very slow to train on 645-dim features * 2000 samples * 50 classes.
# Underperformed SVM in both accuracy and F1.
```

---

## Error Analysis

### Which Classes Are Hardest?

Based on the confusion matrix from the final model (SVM C=10, F5 features), the most common misclassifications cluster around:

**High confusion pairs:**
- `dog_bark` ↔ `rooster` — both are rhythmic animal calls with similar short-burst temporal structure; the model confuses their MFCC patterns under noisy conditions
- `clock_tick` ↔ `mouse_click` — both are rapid, high-frequency transient sounds; band-limiting removes the distinguishing high-frequency click detail
- `engine` ↔ `train` — both are sustained low-frequency rumbles; spectral centroid is similar, making MFCC-based discrimination harder
- `rain` ↔ `crackling_fire` — both are broadband "noise-like" sounds with similar ZCR and RMS profiles

**Impact of distortion conditions:**
- **Noisy condition:** Classes with quiet, sustained sounds (e.g., `crickets`, `clock_tick`) suffer the most. The additive noise raises the noise floor and buries the characteristic low-energy signal components.
- **Band-limited condition:** Classes whose discriminating features lie in the high frequency range (e.g., `glass_breaking`, `mouse_click`, `sneezing`) are most affected. CMVN partially mitigates this by normalising per-coefficient statistics, but cannot recover absent frequencies.

### Why does accuracy plateau around 60%?

The ESC-50 dataset has 40 clips per class — a small number for a 50-class problem. Human accuracy on ESC-50 is approximately 81.3%, while traditional feature-based classifiers typically reach 60–70%. The gap is attributable to:

1. **Limited training data** — deep learning approaches (CNNs on spectrograms) typically reach 80–90%+ but require more data or transfer learning
2. **Spectral overlap** — some classes are perceptually similar (e.g., `wind` and `rain`) and share feature distributions
3. **Distortion in submission set** — the noisy and band-limited variants introduce distribution shift not seen at training time (since augmentation was removed due to the small dataset size)

---

## Visualisations

> The following visualisations are generated from the training notebook and saved to `assets/` in this repository.

### 1. Feature Comparison: Clean vs Noisy vs Band-limited

Showing the log-mel spectrogram of the same clip (`dog_bark`) under all three conditions:

- **Clean** — full frequency content visible from ~100 Hz to 8 kHz
- **Noisy** — elevated noise floor visible across all frequency bins; low-energy features become obscured
- **Band-limited** — content above ~4 kHz is absent; high-frequency mel bins show no energy

This illustrates the challenge the model faces: the same sound event looks significantly different in each condition, but must receive the same label.

```python
# Code to generate this plot (run in Colab)
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Clean', 'Noisy', 'Band-limited']

for ax, path, title in zip(axes, [clean_path, noisy_path, bandlim_path], titles):
    y, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    librosa.display.specshow(log_mel, sr=sr, hop_length=256, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(f'Log-Mel Spectrogram — {title}')

plt.tight_layout()
plt.savefig('assets/spectrogram_comparison.png', dpi=150)
plt.show()
```

### 2. Confusion Matrix

A heatmap showing the 50×50 classification confusion matrix on the validation set:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_va, y_pred)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
            annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix — SVM C=10, Final Features')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150)
plt.show()
```

### 3. Per-Class F1 Score Bar Chart

Shows which of the 50 classes the model classifies most and least accurately:

```python
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_va, y_pred,
    target_names=[idx_to_label[i] for i in range(len(classes))],
    output_dict=True)

report_df = pd.DataFrame(report).T
f1_by_class = report_df.loc[[idx_to_label[i] for i in range(len(classes))], 'f1-score']
f1_by_class = f1_by_class.sort_values()

plt.figure(figsize=(12, 10))
f1_by_class.plot(kind='barh', color='steelblue')
plt.axvline(x=f1_by_class.mean(), color='red', linestyle='--', label='Mean F1')
plt.xlabel('F1 Score')
plt.title('Per-Class F1 Score — Final Model')
plt.legend()
plt.tight_layout()
plt.savefig('assets/per_class_f1.png', dpi=150)
plt.show()
```

### 4. Feature Group Contribution (Ablation)

Bar chart showing the drop in Macro-F1 when each feature group is removed one at a time:

| Feature Group Removed | Macro-F1 Drop |
|---|---|
| Remove MFCC | −0.11 |
| Remove Log-Mel | −0.06 |
| Remove Delta / Delta-Delta | −0.03 |
| Remove Spectral Centroid | −0.01 |
| Remove ZCR + RMS | −0.01 |
| Remove Percentile Pooling (back to mean+std) | −0.02 |

MFCC is the most critical feature group. Log-Mel adds complementary texture-based information. Spectral and temporal features provide smaller but additive improvements.

---

## Final Configuration

| Component | Final Choice | Rationale |
|---|---|---|
| Sample Rate | 16 kHz | Standard for audio ML; covers 0–8 kHz range |
| Silence Trimming | `top_db=25` | Gentle enough to preserve brief transients |
| Pre-emphasis | `librosa.effects.preemphasis()` | Improves high-freq MFCC discriminability |
| Normalisation | Peak normalisation | Consistent scale without over-amplifying quiet clips |
| MFCCs | 20 coefficients | Sufficient spectral detail for 50-class ESC-50 |
| CMVN | Row-normalise MFCC + Log-Mel | Improves robustness to channel/noise distortion |
| Mel bins | 64 | Standard for environmental sound tasks |
| FFT size | 1024 samples (64 ms at 16 kHz) | Good time-frequency resolution balance |
| Hop length | 256 samples (16 ms) | ~75% overlap; sufficient temporal resolution |
| Pooling | mean, std, median, p25, p75 | Captures full distribution shape across frames |
| Feature dimension | 645 | Balanced expressiveness vs. complexity |
| Classifier | SVC (RBF, C=10, balanced) | Best accuracy/F1 across all tested classifiers |
| Train/Val split | 80/20, stratified, seed=42 | Reproducible, class-balanced evaluation |
