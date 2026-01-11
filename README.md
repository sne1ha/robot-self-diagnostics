# Teaching Robots to Predict Their Own Deaths  
### Machine Learning for Self-Diagnostic Manufacturing Robots

## Academic Context

This project was completed as the **final course project** for  
**Case Studies in Machine Learning (CSML)**  
**The University of Texas at Austin**

**Full paper:**  
[Teaching Robots to Predict Their Own Deaths: ML for Self-Diagnostic Manufacturing](https://drive.google.com/file/d/1MuBJ-g9E_FpObhpwWPIR-urV6yzhi7Io/view?usp=sharing)

## Overview

Industrial robots operate under continuous mechanical stress, and unexpected failures can halt entire production lines—costing manufacturers millions of dollars per hour. Traditional predictive maintenance systems rely on additional sensors (vibration, force, vision), increasing cost and system complexity.

This project explores a **sensor-free alternative**:

> **Can industrial robot degradation and faults be detected using only controller-level kinematic data?**

Using joint positions, velocities, and motor currents from a UR3 robot, I evaluated machine learning models for anomaly detection and fault classification **without any additional hardware**.


---

## Key Contributions

- Shows that **controller-only kinematic data contains meaningful fault information**
- Evaluates **real-world vs synthetic degradation signals**
- Compares:
  - Feature-based vs sequence-based models
  - Unsupervised vs supervised learning
  - Classical ML vs deep learning

---

## Datasets

### AURSAD (Real-World Dataset)
- UR3 screwdriving assembly executions
- Real mechanical faults
- Rare and subtle fault signatures
- Downsampled to **12.5 Hz**
- Strong class imbalance

### Hybrid Dataset (Synthetic + Realistic Faults)
- Combines AURSAD-like faults with controlled synthetic degradations
- Maintains full **100 Hz** sampling rate
- Synthetic degradations:
  - Bearing wear
  - Gear backlash
  - Motor degradation
  - Increased friction

| Property | AURSAD | Hybrid |
|--------|--------|--------|
| Trajectories | 4,091 | 1,350 |
| Sampling Rate | 12.5 Hz | 100 Hz |
| Joints Used | 6 | 3 |
| Channels | 18 | 18 |

---

## Feature Engineering

Each robot trajectory is converted into a fixed-length vector of **281 engineered features** derived exclusively from controller-level signals.

### Feature Categories

- **Statistical**  
  - Mean: Average value of a signal. Shows the typical level of movement or force.
  - Standard deviation (std): How much the signal varies around the mean. High std = more variability.
  - Variance: Just the square of std; another measure of variability.
  - Skewness: Measures asymmetry. Is the signal biased toward high or low values?
  - Kurtosis: Measures “peakedness” or whether there are extreme spikes in the signal.
  - RMS (Root Mean Square): Measures the overall magnitude of the signal, similar to energy.

Simple, computationally cheap, and give a baseline sense of how the robot is moving or applying forces.
Even if sampling is low, stats can reveal abnormalities like jitter, drift, or excessive vibrations.

- **Frequency**  
  - FFT (Fast Fourier Transform): Converts a time signal into frequencies. Helps find repeating patterns or oscillations.
  - Welch PSD (Power Spectral Density): Estimates how energy is distributed across frequencies, smoother than plain FFT.
  - Dominant frequencies: The most “active” frequencies—e.g., robot joint shaking at 5 Hz.
  - Band power ratios: Energy in different frequency bands relative to each other. Can indicate whether movement is smooth or jerky.

Mechanical faults often show up as vibrations or oscillations.
Frequency analysis captures hidden periodic issues that time-domain stats might miss.

- **Wavelet**  
  - Discrete Wavelet Transform (db4): Breaks the signal into multiple scales (coarse and fine).
  - Energy features: Measures how much “activity” occurs at each scale.

Robot signals can have sudden changes or spikes, not just steady patterns.
Wavelets capture both time and frequency information, unlike FFT which only gives frequency.

- **Temporal**  
  - Autocorrelation: How much the signal resembles itself over time, detects repeating patterns.
  - Linear trend slope: Detects gradual drift in signals (e.g., motor overheating slowly).
  - Zero-crossing rate: How often the signal crosses zero—captures rapid oscillations.
  - Peak counts: Number of sharp peaks—can indicate jerks or impacts.

Capture temporal patterns and trends in robot behavior, important for detecting early anomalies.
  
- **Cross-Joint**  
  - Velocity correlations: Measures if two joints move together or independently.
  - Coordination metrics: Checks if joint movements follow expected patterns.

Faults often disrupt coordination between joints (e.g., one motor lags behind).
Helps detect system-level issues, not just local sensor anomalies

Feature-based representations preserve fault information even at low sampling rates.

---

## Models

### Unsupervised Anomaly Detection

Trained using **normal trajectories only**.

- **Isolation Forest**
- **LSTM Autoencoder**

| Dataset | Model | F1 | Precision | Recall |
|-------|------|----|----------|--------|
| Hybrid | Isolation Forest | 0.67 | 1.00 | 0.50 |
| Hybrid | LSTM Autoencoder | 0.70 | 1.00 | 0.54 |
| AURSAD | Isolation Forest | 0.67 | 1.00 | 0.50 |
| AURSAD | LSTM Autoencoder | 0.31 | 1.00 | 0.18 |

---

### Supervised Fault Classification

Multi-class classification using engineered trajectory features.

- **Random Forest**
- **XGBoost**
- Class imbalance handled with:
  - SMOTE oversampling
  - Class-weighted loss

| Dataset | Model | Accuracy | Macro F1 |
|-------|------|----------|----------|
| Hybrid | Random Forest | 0.95 | 0.97 |
| Hybrid | XGBoost | 0.97 | 0.98 |
| AURSAD | Random Forest | 0.83 | 0.69 |
| AURSAD | XGBoost | 0.83 | 0.71 |

---

### Transfer Learning (Sequence Models)

- LSTM encoder pre-trained on normal trajectories (reconstruction objective)
- Fine-tuned for multi-class fault classification

---

## Experimental Pipeline

Raw Robot Trajectories
↓
Preprocessing & Normalization
↓
Feature Engineering (281 features)
↓
Unsupervised Detection / Supervised Classification
↓
Evaluation (F1, Precision, Recall)

## Tech Stack

- Python
- NumPy, SciPy
- scikit-learn
- XGBoost
- PyTorch
- imbalanced-learn
- matplotlib


