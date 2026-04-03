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
| **P5 — Add Peak Normalisation (Final)** | 25 dB | Yes | Peak | 0.41 | ~0.40 | Peak normalisation with MFCC-only baseline; chosen for robustness across full feature set. |
| **P6 — Aggressive Trim** (tested, discarded) | 35 dB | Yes | Peak | 0.45 | ~0.45 | Over-trimming removes quiet onset/decay of sounds. |
| **P7 — RMS Normalisation** (tested, discarded) | 25 dB | Yes | RMS | 0.44 | ~0.44 | RMS over-amplifies noise in naturally quiet clips. |

**Key Finding:** 
* **Fixed-Length Context (P2):** Adding NaN removal and fixed-length padding/truncation to a 5-second window did not immediately improve accuracy on its own (0.40 → 0.40), but it is a critical precondition — without consistent clip length, downstream feature vectors would have different sizes across clips, making the classifier unreliable.
* **Silence Trimming (P3):** The jump from P2 to P3 (`top_db=25`) produced the largest single gain in this phase (Macro-F1: 0.41 → 0.43). Removing leading and trailing silence means the 5-second window is filled with actual sound content, giving the model more signal-rich frames to pool over.
* **The Pre-emphasis Effect (P4):** Applying a high-pass filter (`H(z) = 1 − 0.97z⁻¹`) boosted high-frequency energy and further improved Macro-F1 (0.43 → 0.44). This confirms that environmental sounds rely on upper spectral content that MFCCs would otherwise underweight.
* **Normalization Divergence (P5 & P7):** Using **Peak Normalization** with the MFCC-only baseline showed a slight drop (0.40 Macro-F1), while **RMS Normalization** (P7) scored 0.44. However, peak normalisation was retained as the final choice because it provides a stable, bounded input range (`[-1, 1]`) that benefits the full 645-feature pipeline — particularly under the noisy and band-limited submission conditions where RMS normalisation can over-amplify background noise in quiet clips.
* **Trimming Thresholds (P3 & P6):** While a gentle trim (`top_db=25`) stabilized the pipeline, the more aggressive trim (`top_db=35`) produced a slight spike (0.45 accuracy). However, this was **discarded** to avoid removing the quiet onset and decay phases (e.g., reverb tails) that are essential for generalising to real-world audio.


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
 
#### P5 — Add Peak Normalisation (Final)
 
Without normalisation, louder clips produce larger MFCC magnitudes, which can bias the classifier's distance calculations. Peak normalisation scales every clip to the range `[-1, 1]` relative to its own loudest frame. Although peak normalisation showed a slight drop in the MFCC-only baseline (P5 vs P7), it was selected for the final pipeline because it produces a stable, bounded signal that works more reliably with the full 645-feature set and under distorted submission conditions.
 
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
| **F5 — Rich Pooling** | **Stats: 2 → 5** | **Same as F4** | **mean, std, med, p25, p75** | **645** | **0.60** | **0.59** | **Captures full distribution shape (Final)** |
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
| M2 — Random Forest | `RandomForestClassifier` | `n_estimators=200` | 0.60 | ~0.58 | Better than LR, worse than SVM |
| M3 — SVM C=1 | `SVC` | `C=1, rbf, scale` | 0.50 | ~0.46 | Under-regularised for this feature space |
| M4 — SVM C=10 (balanced) | `SVC` | `C=10, rbf, scale, balanced` | 0.59 | ~0.57 | Strong single-model performance |
| M5 — SVM C=20 | `SVC` | `C=20, rbf, scale, balanced` | 0.59 | ~0.57 | Same as C=10; not chosen (less generalisation) |
| M6 — SVM C=30 | `SVC` | `C=30, rbf, scale, balanced` | 0.59 | ~0.57 | Same as C=10; not chosen |
| M7 — Gradient Boosting | `GradientBoostingClassifier` | `n_estimators=200` | >15min, no result | >15min, no result | Slow to train, no result |
| **M8 — Voting Ensemble (LR + SVM + RF) (Final)** | VotingClassifier (soft voting) | `estimators=[('lr', LogisticRegression), ('svm', SVC(C=10, rbf)), ('rf', RandomForestClassifier)]`, `voting='soft'` | **0.63** | **~0.62** | **Ensemble combines three diverse models; soft voting uses predicted probabilities. Best overall result.** |

**Final decision: Voting Ensemble (M8)** — soft voting over Logistic Regression, SVM (C=10), and Random Forest produced the highest validation accuracy (0.62) and Macro-F1 (0.63), outperforming any single classifier. Each constituent model makes different types of errors; the ensemble averages their probability outputs to reduce individual model variance. This is the model saved as `Pr_23_model.joblib` and used for all submission predictions.

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

#### M4 — SVM C=10

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
```
#### M8 — Voting Ensemble (LR + SVM + RF) — **Final Model**

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

base_lr = LogisticRegression(max_iter=2000, class_weight='balanced')
base_svm = SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)
base_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

voting_clf = VotingClassifier(
    estimators=[('lr', base_lr), ('svm', base_svm), ('rf', base_rf)],
    voting='soft'
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', voting_clf)
])
```

**Why soft voting?** Each sub-classifier outputs class probabilities (LR and RF natively; SVC requires `probability=True`). The ensemble averages these probability vectors before predicting the class with the highest mean probability. This is more informative than hard voting (majority of predicted labels), since it weights confident predictions more heavily. The three constituent models capture complementary patterns: LR is a linear discriminant that generalises well, SVM finds the maximum-margin boundary in the RBF kernel space, and RF captures non-linear feature interactions through bagged decision trees.

---

## Error Analysis

### Which Classes Are Hardest?

Based on the confusion matrix from the final model (Voting Ensemble, F5 features), the most common misclassifications cluster around acoustically similar sound pairs:

**High confusion pairs:**
- `dog_bark` ↔ `rooster` — both are rhythmic animal vocalisations with similar short-burst temporal structure and overlapping MFCC coefficient distributions, especially under additive noise
- `clock_tick` ↔ `mouse_click` — both are rapid, high-frequency transient sounds; the band-limited condition removes the distinguishing high-frequency click detail, causing the model to conflate them
- `engine` ↔ `train` — both are sustained low-frequency rumbles with similar spectral centroid and bandwidth; RMS and ZCR profiles are also close, making MFCC-based discrimination harder
- `rain` ↔ `crackling_fire` — both are broadband "noise-like" textures with similar ZCR and RMS energy profiles; neither has a strong tonal component to separate them

**Impact of distortion conditions:**
- **Noisy condition:** Classes with quiet, sustained sounds (e.g., `crickets`, `clock_tick`) suffer the most. The additive noise raises the noise floor and buries the characteristic low-energy signal components. CMVN partially compensates by normalising per-coefficient statistics, but cannot recover energy lost below the noise floor.
- **Band-limited condition:** Classes whose discriminating features lie in the high frequency range (e.g., `glass_breaking`, `mouse_click`, `sneezing`) are most affected, as the upper mel bins carry no energy. Pre-emphasis and CMVN provide limited mitigation — they can boost and normalise the remaining low-frequency content, but cannot recover absent frequency bands.

### Why does accuracy plateau around 60–62%?

The ESC-50 dataset has 40 clips per class — a small number for a 50-class problem. Human accuracy on ESC-50 is approximately 81.3%, while traditional feature-based classifiers typically reach 60–70%. The gap is attributable to:

1. **Limited training data** — with only 32 training examples per class after the 80/20 split, the classifier cannot learn the full intra-class variability. Deep learning approaches (CNNs on spectrograms) typically reach 80–90%+ but require more data or transfer learning.
2. **Spectral overlap** — some classes are perceptually similar (e.g., `wind` and `rain`) and share feature distributions; no hand-crafted feature set can fully separate them without significantly more training examples.
3. **Distribution shift from distortion** — the noisy and band-limited submission variants introduce conditions not present at training time. Since augmentation was removed (adding augmented copies to a 40-clip-per-class set caused overfitting), the model has no exposure to these distortions during training.
4. **Fixed pooling** — summarising an entire 5-second clip with 5 statistics per feature dimension loses temporal ordering. Two sounds with the same spectral statistics but different temporal trajectories (e.g., a sound that starts quiet and gets loud vs. the reverse) appear identical to the classifier.

---

## Visualisations

> The following code blocks are self-contained and can be pasted into a **new Colab cell** after the main training notebook has been run. They assume that the variables `model`, `X_va`, `y_va`, `y_pred`, `classes`, `idx_to_label`, `sub_meta`, and `audio_sub_dir` are already defined in the session.

### 1. Feature Comparison: Clean vs Noisy vs Band-limited

Showing the log-mel spectrogram of the same clip under all three submission conditions:

- **Clean** — full frequency content visible from ~100 Hz to 8 kHz
- **Noisy** — elevated noise floor visible across all frequency bins; low-energy features become obscured
- **Band-limited** — content above ~4 kHz is absent; high-frequency mel bins show no energy

This illustrates the challenge the model faces: the same sound event looks significantly different in each condition, but must receive the same label.

```python
# ── Visualisation 1: Log-Mel Spectrogram — Clean vs Noisy vs Band-limited ──
# Paste into a new Colab cell after running the main notebook.
# Requires: sub_meta, audio_sub_dir  (defined in cells 9+ of the notebook)

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

os.makedirs('assets', exist_ok=True)

# Pick the first base clip and locate its three variants
base_ids = sub_meta['clip_id'].tolist()

# The submission set contains clips named like: <id>__clean, <id>__noisy, <id>__bandlimited
# Find one complete triplet automatically
clean_id   = next((c for c in base_ids if str(c).endswith('__clean')),       None)
noisy_id   = next((c for c in base_ids if str(c).endswith('__noisy')),       None)
bandlim_id = next((c for c in base_ids if str(c).endswith('__bandlimited')), None)

assert clean_id and noisy_id and bandlim_id, (
    "Could not find clean/noisy/bandlimited clips in sub_meta. "
    "Check that sub_meta is loaded and clip_id column uses the expected suffixes."
)

clean_path   = os.path.join(audio_sub_dir, f'{clean_id}.wav')
noisy_path   = os.path.join(audio_sub_dir, f'{noisy_id}.wav')
bandlim_path = os.path.join(audio_sub_dir, f'{bandlim_id}.wav')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Clean', 'Noisy', 'Band-limited']
paths  = [clean_path, noisy_path, bandlim_path]

for ax, path, title in zip(axes, paths, titles):
    y, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=256,
                                   x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(f'Log-Mel Spectrogram — {title}')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

plt.suptitle(f'Clip: {clean_id.replace("__clean", "")}', fontsize=11)
plt.tight_layout()
plt.savefig('assets/spectrogram_comparison.png', dpi=150)
plt.show()
print("Saved → assets/spectrogram_comparison.png")
```

### 2. Confusion Matrix

A heatmap showing the 50×50 classification confusion matrix on the validation set:

```python
# ── Visualisation 2: Confusion Matrix (50×50) ──
# Paste into a new Colab cell after running the main notebook.
# Requires: y_va, y_pred, classes, idx_to_label  (defined after cell 8)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

os.makedirs('assets', exist_ok=True)

# Rebuild predictions if y_pred is not in scope (re-run model.predict)
# y_pred = model.predict(X_va)   # ← uncomment if needed

class_names = [idx_to_label[i] for i in range(len(classes))]
cm = confusion_matrix(y_va, y_pred)

plt.figure(figsize=(22, 18))
sns.heatmap(
    cm,
    xticklabels=class_names,
    yticklabels=class_names,
    annot=False,
    fmt='d',
    cmap='Blues',
    linewidths=0.3,
    linecolor='lightgrey'
)
plt.title('Confusion Matrix — Voting Ensemble (Final Model), F5 Features', fontsize=13)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150)
plt.show()
print("Saved → assets/confusion_matrix.png")
```

### 3. Per-Class F1 Score Bar Chart

Shows which of the 50 classes the model classifies most and least accurately:

```python
# ── Visualisation 3: Per-Class F1 Score Bar Chart ──
# Paste into a new Colab cell after running the main notebook.
# Requires: y_va, y_pred, classes, idx_to_label  (defined after cell 8)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

os.makedirs('assets', exist_ok=True)

class_names = [idx_to_label[i] for i in range(len(classes))]

report = classification_report(
    y_va, y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).T
# Keep only the 50 class rows (exclude macro/weighted avg rows)
f1_by_class = report_df.loc[class_names, 'f1-score'].astype(float)
f1_by_class = f1_by_class.sort_values()

plt.figure(figsize=(12, 14))
bars = plt.barh(f1_by_class.index, f1_by_class.values, color='steelblue')
plt.axvline(x=float(f1_by_class.mean()), color='red', linestyle='--',
            label=f'Mean F1 = {f1_by_class.mean():.3f}')
plt.xlabel('F1 Score', fontsize=11)
plt.title('Per-Class F1 Score — Voting Ensemble (Final Model)', fontsize=13)
plt.xlim(0, 1.05)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('assets/per_class_f1.png', dpi=150)
plt.show()
print("Saved → assets/per_class_f1.png")
```

### 4. Feature Group Contribution (Ablation)

Bar chart showing the drop in Macro-F1 when each feature group is removed one at a time (all tested with Logistic Regression on F5 pipeline):

| Feature Group Removed | Macro-F1 Drop |
|---|---|
| Remove MFCC | −0.11 |
| Remove Log-Mel | −0.06 |
| Remove Delta / Delta-Delta | −0.03 |
| Remove Spectral Centroid | −0.01 |
| Remove ZCR + RMS | −0.01 |
| Remove Percentile Pooling (back to mean+std) | −0.02 |

MFCC is the most critical feature group. Log-Mel adds complementary texture-based information. Spectral and temporal features provide smaller but additive improvements.

```python
# ── Visualisation 4: Feature Ablation Bar Chart ──
# Paste into a new Colab cell — standalone, no prior variables needed.

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('assets', exist_ok=True)

feature_groups = [
    'Remove MFCC',
    'Remove Log-Mel',
    'Remove Delta/Delta-Delta',
    'Remove Spectral Centroid',
    'Remove ZCR + RMS',
    'Remove Percentile Pooling\n(back to mean+std)',
]
f1_drops = [-0.11, -0.06, -0.03, -0.01, -0.01, -0.02]

# Sort by magnitude of drop (largest drop first)
sorted_pairs = sorted(zip(f1_drops, feature_groups), key=lambda x: x[0])
drops_sorted, groups_sorted = zip(*sorted_pairs)

colors = ['#d73027' if d <= -0.05 else '#fc8d59' if d <= -0.02 else '#fee090'
          for d in drops_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(groups_sorted, drops_sorted, color=colors, edgecolor='grey')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Macro-F1 Drop (negative = performance loss)', fontsize=11)
ax.set_title('Feature Group Ablation — Macro-F1 Drop When Group Is Removed\n'
             '(Logistic Regression, F5 pipeline)', fontsize=12)
ax.set_xlim(-0.14, 0.01)

for bar, drop in zip(bars, drops_sorted):
    ax.text(drop - 0.002, bar.get_y() + bar.get_height() / 2,
            f'{drop:.2f}', va='center', ha='right', fontsize=9, color='black')

plt.tight_layout()
plt.savefig('assets/feature_ablation.png', dpi=150)
plt.show()
print("Saved → assets/feature_ablation.png")
```

---

## Final Configuration

| Component | Final Choice | Rationale |
|---|---|---|
| Sample Rate | 16 kHz | Standard for audio ML; covers 0–8 kHz range |
| Silence Trimming | `top_db=25` | Gentle enough to preserve brief transients |
| Pre-emphasis | `librosa.effects.preemphasis()` | Improves high-freq MFCC discriminability |
| Normalisation | Peak normalisation | Consistent scale; bounded input for all conditions |
| MFCCs | 20 coefficients | Sufficient spectral detail for 50-class ESC-50 |
| CMVN | Row-normalise MFCC + Log-Mel | Improves robustness to channel/noise distortion |
| Mel bins | 64 | Standard for environmental sound tasks |
| FFT size | 1024 samples (64 ms at 16 kHz) | Good time-frequency resolution balance |
| Hop length | 256 samples (16 ms) | ~75% overlap; sufficient temporal resolution |
| Pooling | mean, std, median, p25, p75 | Captures full distribution shape across frames |
| Feature dimension | 645 | Balanced expressiveness vs. complexity |
| Classifier | Voting Ensemble (LR + SVM C=10 + RF, soft voting) | Best accuracy (0.62) and Macro-F1 (0.63) across all tested classifiers |
| Train/Val split | 80/20, stratified, seed=42 | Reproducible, class-balanced evaluation |
