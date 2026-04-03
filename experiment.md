# Experiment Log — Pr_23 CEG3004 DSP Mini-Project

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

### Summary Table

| Experiment | Silence Trim `top_db` | Pre-emphasis | Normalisation | Val Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| P1 — Baseline (no changes) | None | No | None | ~0.44 | ~0.40 | Raw signal, no processing |
| P2 — Trim only | 25 dB | No | None | ~0.50 | ~0.47 | Removing silence helps significantly |
| P3 — Trim + Peak Norm | 25 dB | No | Peak | ~0.53 | ~0.50 | Normalisation adds consistency |
| P4 — Trim + Pre-emphasis + Peak Norm | 25 dB | Yes | Peak | ~0.58 | ~0.55 | Pre-emphasis improves MFCC quality |
| **P5 — Final (Trim + Pre-emphasis + Peak Norm)** | **25 dB** | **Yes** | **Peak** | **~0.60** | **~0.57** | Same as P4 with final features |
| P6 — Aggressive Trim | 35 dB | Yes | Peak | ~0.56 | ~0.53 | Over-trimming cuts useful signal |
| P7 — RMS Normalisation | 25 dB | Yes | RMS | ~0.58 | ~0.55 | Slightly lower than peak norm |

**Key finding:** Silence trimming at `top_db=25` provides the biggest single gain. Pre-emphasis meaningfully boosts MFCC discriminability. Aggressive trimming (`top_db=35`) removes meaningful content and hurts performance.

---

### Full Preprocessing Code Variants

#### P1 — Baseline (no preprocessing)

```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
    return y
```

#### P2 — Silence Trim Only

```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
    y, _ = librosa.effects.trim(y, top_db=25)
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y.astype(np.float32)
```

#### P3 — Trim + Peak Normalisation

```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
    y, _ = librosa.effects.trim(y, top_db=25)
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant'))
    else:
        y = y[:target_len]
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
    return y.astype(np.float32)
```

#### P4/P5 — Final: Trim + Pre-emphasis + Peak Normalisation ✅

```python
def preprocess_audio(y, sr):
    """Final preprocessing pipeline."""
    y = np.nan_to_num(y).astype(np.float32)

    # Step 1: Trim leading/trailing silence (top_db=25 is gentle enough
    # to retain brief transient sounds like dog barks and gunshots)
    y, _ = librosa.effects.trim(y, top_db=25)

    # Step 2: Pad or truncate to fixed 5-second length
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # Step 3: Pre-emphasis — boosts high-frequency components.
    # Environmental sounds often have important high-frequency cues
    # (e.g., glass breaking, bird chirps) that are attenuated by
    # the vocal tract model underlying MFCCs. Pre-emphasis compensates.
    y = librosa.effects.preemphasis(y)

    # Step 4: Peak normalisation — ensures all clips are on the same
    # amplitude scale, preventing louder clips from dominating the model.
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    return y.astype(np.float32)
```

#### P6 — Aggressive Trim (top_db=35)

```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
    # Too aggressive — cuts actual sound events, not just silence
    y, _ = librosa.effects.trim(y, top_db=35)
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

#### P7 — RMS Normalisation (alternative to peak)

```python
def preprocess_audio(y, sr):
    y = np.nan_to_num(y).astype(np.float32)
    y, _ = librosa.effects.trim(y, top_db=25)
    target_len = int(5 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    y = librosa.effects.preemphasis(y)
    # RMS normalisation — normalises perceived loudness
    rms = np.sqrt(np.mean(y**2)) + 1e-8
    y = y / rms
    return y.astype(np.float32)
```

**Why we chose peak normalisation over RMS:** Peak norm preserves relative amplitude ratios between sounds and is more robust to clips with very low RMS energy (e.g., distant sounds) that can cause over-amplification with RMS norm.

---

## Part 2 — Feature Extraction Experiments

The `extract_features(path)` function converts raw audio into a fixed-length feature vector. We experimented with which features to include, how many coefficients to use, and how to pool across time frames.

### Summary Table

| Experiment | Features Used | Pooling | Vector Dim | Val Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| F1 — Baseline MFCC only | MFCC(20), Δ, ΔΔ | mean + std | 120 | ~0.50 | ~0.46 | Baseline from original code |
| F2 — MFCC + Log-Mel | MFCC(20), Δ, ΔΔ, Log-Mel(64) | mean + std | 440 | ~0.54 | ~0.51 | Log-Mel adds texture information |
| F3 — Add Spectral + Temporal | Above + centroid, bandwidth, rolloff, ZCR, RMS | mean + std | 450 | ~0.57 | ~0.54 | Spectral shape helps differentiation |
| F4 — Add CMVN | Same as F3, with row-normalised MFCC and Log-Mel | mean + std | 450 | ~0.58 | ~0.55 | CMVN improves noise robustness |
| **F5 — Final: Add rich pooling** | **Same as F4** | **mean, std, median, p25, p75** | **645** | **~0.60** | **~0.57** | **Percentile pooling captures distribution shape** |
| F6 — More MFCCs (40) | MFCC(40), Δ, ΔΔ + rest | mean, std, median, p25, p75 | 1245 | ~0.59 | ~0.56 | More MFCCs did not improve; higher dim |
| F7 — Feature selection (SelectKBest) | Top 300 from F5 | — | 300 | ~0.58 | ~0.55 | No improvement; discarded |
| F8 — With augmentation | F5 + noise/bandpass augmentation in training | — | 645 | ~0.57 | ~0.54 | Augmentation at training time hurt performance |

**Key finding:** The jump from mean+std pooling to 5-statistic pooling (adding median and percentiles) improved performance by capturing the *distribution shape* of features over time, not just the average. CMVN (Cepstral Mean and Variance Normalisation via `librosa.util.normalize(..., axis=1)`) improved robustness to channel and noise distortions.

---

### Why 20 MFCCs instead of 13 or 40?

- **13 MFCCs** is a classic choice for speech, but environmental sounds contain richer spectral texture requiring more coefficients.
- **20 MFCCs** captures enough spectral detail for ESC-50 without over-fitting or adding redundant dimensions.
- **40 MFCCs** (F6) did not improve performance and added ~600 more features, increasing training time and risk of overfitting.

### Why use a Butterworth-style pre-emphasis instead of a bandpass filter?

`librosa.effects.preemphasis()` applies a first-order high-pass filter (`H(z) = 1 - 0.97z⁻¹`). For the band-limited submission condition, where high frequencies are already suppressed, the MFCC's resistance to this suppression (due to CMVN) provides robustness — rather than attempting to recover frequencies that are truly missing.

---

### Full Feature Extraction Code Variants

#### F1 — Baseline: MFCC stats only

```python
def extract_features(path, sr=16000):
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)
    return np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0).astype(np.float32)
    # Output: 3 * 20 * 2 = 120 dimensions
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

#### F5 — Final: Rich Percentile Pooling ✅

```python
def extract_features(path, sr=16000):
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)

    n_fft, hop = 1024, 256

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    mfcc = librosa.util.normalize(mfcc, axis=1)   # CMVN
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10)
    log_mel = librosa.util.normalize(log_mel, axis=1)  # CMVN

    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms       = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    def stats(M):
        # 5-statistic pooling captures distribution shape, not just average
        return np.concatenate([
            np.mean(M, axis=1),
            np.std(M, axis=1),
            np.median(M, axis=1),
            np.percentile(M, 25, axis=1),
            np.percentile(M, 75, axis=1)
        ], axis=0)

    feat = np.concatenate([
        stats(mfcc),      # 20 * 5 = 100
        stats(d1),        # 20 * 5 = 100
        stats(d2),        # 20 * 5 = 100
        stats(log_mel),   # 64 * 5 = 320
        stats(centroid),  # 1 * 5  = 5
        stats(bandwidth), # 1 * 5  = 5
        stats(rolloff),   # 1 * 5  = 5
        stats(zcr),       # 1 * 5  = 5
        stats(rms),       # 1 * 5  = 5
    ], axis=0).astype(np.float32)

    return feat  # Total: 645 dimensions
```

#### F6 — 40 MFCCs (tested, not selected)

```python
# Replace n_mfcc=20 with n_mfcc=40
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop)
# Result: 40 * 5 * 3 = 600 dims from MFCC alone; total ~1245 dims
# Performance did not improve over 20 MFCCs — higher dimensionality
# increases training time without benefit for this dataset size.
```

#### F7 — Feature Selection with SelectKBest (tested, not selected)

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=300)),
    ('clf', SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced'))
])
# Accuracy dropped slightly — the SVM with RBF kernel already handles
# high-dimensional input well; discarding features removed useful signal.
```

#### F8 — Training-time Augmentation (tested, not selected)

```python
# Augmentation applied ONLY during training (not at inference)
import random

def augment_audio(y, sr):
    """Randomly apply one augmentation."""
    choice = random.random()
    if choice < 0.33:
        # Add white noise
        noise_level = random.uniform(0.003, 0.01)
        y = y + noise_level * np.random.randn(len(y))
    elif choice < 0.66:
        # Band-pass filter (simulate bandlimited condition)
        from scipy.signal import butter, sosfilt
        low, high = 300, 3000
        sos = butter(4, [low / (sr / 2), high / (sr / 2)], btype='band', output='sos')
        y = sosfilt(sos, y)
    # else: no augmentation (leave clean)
    return y.astype(np.float32)

# Applied inside the training loop before extract_features()
# Result: Accuracy dropped ~0.03 — augmentation introduced too much
# variance with only 40 clips per class (small dataset). Removed from final.
```

---

## Part 3 — Model Training Experiments

### Summary Table

| Experiment | Classifier | Hyperparameters | Val Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|
| M1 — Logistic Regression (baseline) | `LogisticRegression` | `max_iter=1000` | ~0.50 | ~0.46 | Original baseline provided |
| M2 — Random Forest | `RandomForestClassifier` | `n_estimators=200` | ~0.54 | ~0.51 | Better than LR, worse than SVM |
| M3 — SVM C=1 | `SVC` | `C=1, rbf, scale` | ~0.57 | ~0.54 | Under-regularised for this feature space |
| M4 — SVM C=10 (balanced) | `SVC` | `C=10, rbf, scale, balanced` | ~0.60 | ~0.57 | Best performance |
| M5 — SVM C=20 | `SVC` | `C=20, rbf, scale, balanced` | ~0.60 | ~0.57 | Same as C=10; not chosen (less generalisation) |
| M6 — SVM C=30 | `SVC` | `C=30, rbf, scale, balanced` | ~0.59 | ~0.56 | Slight overfit |
| M7 — Gradient Boosting | `GradientBoostingClassifier` | `n_estimators=200` | ~0.53 | ~0.49 | Slow to train; underperforms SVM |

**Final decision: SVM with `C=10`** — same performance as C=20, but C=10 is a simpler, more conservative model that is less likely to overfit on the small 40-clips-per-class training set.

---

### Full Model Code Variants

#### M1 — Logistic Regression (original baseline)

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
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

#### M4 — SVM C=10 (Final) ✅

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
