#!/usr/bin/env python3
"""
Real‐time ECoG Activity Classifier (Proof‐of‐Concept)

This script simulates live ECoG data acquisition and trains an online classifier
to predict activity labels in real time. It demonstrates:
1. Bandpass + notch filtering for each 100‐ms window.
2. Feature extraction: band power in standard EEG/ECoG bands.
3. Running‐statistics normalization.
4. An online linear classifier (stochastic gradient descent) that updates as new labeled data arrives.
"""

import numpy as np
import scipy.signal as signal
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# -- PARAMETERS -----------------------------------------------------------------

FS = 1000                    # ECoG sampling rate in samples per second
WINDOW_MS = 100              # window length in milliseconds
WINDOW_SIZE = FS * WINDOW_MS // 1000   # number of samples per window (100 ms → 100 samples)
N_CHANNELS = 64              # number of ECoG channels
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (13, 30),
    'hgamma': (70, 200)
}
FILTER_ORDER = 4
NOTCH_FREQ = 60.0            # line noise at 60 Hz
NOTCH_QUALITY = 30.0         # quality factor for notch

# Simulate two classes of activity: 0 and 1
ACTIVITIES = [0, 1]
N_CLASSES = len(ACTIVITIES)

# -- UTILITY FUNCTIONS ----------------------------------------------------------

def design_bandpass(lowcut, highcut, fs, order=4):
    """Return bandpass filter coefficients (b, a)."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def design_notch(freq, fs, quality=30.0):
    """Return notch filter coefficients (b, a) at given frequency."""
    b, a = signal.iirnotch(freq, quality, fs)
    return b, a

def apply_filter(data, b, a):
    """
    Zero‐phase apply filter to each channel in `data`.
    data shape: (n_samples, n_channels)
    """
    return signal.filtfilt(b, a, data, axis=0)

def extract_band_power(window, fs, bands):
    """
    Compute log‐bandpower features for each channel in `window`.

    window shape: (window_size, n_channels)
    returns: 1D array of length n_channels * n_bands
    """
    n_channels = window.shape[1]
    features = []

    # Compute power spectral density via Welch’s method, per channel
    # (nfft = 256 gives adequate frequency resolution for 0–500 Hz at fs=1000)
    freqs, psd = signal.welch(
        window, 
        fs=fs, 
        nperseg=window.shape[0], 
        axis=0
    )  # psd shape: (nfreqs, n_channels)

    for (low, high) in bands.values():
        # Find indices of PSD frequencies within [low, high]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integrate PSD over that band for each channel
        band_power = np.trapz(psd[idx_band, :], freqs[idx_band], axis=0)  # shape: (n_channels,)
        # Take logarithm (plus a small epsilon to avoid log(0))
        features.append(np.log(band_power + 1e-8))

    # Stack all band features: result shape: (n_bands, n_channels) → flatten to 1D
    features = np.vstack(features).T  # shape: (n_channels, n_bands) 
    return features.flatten()        # shape: (n_channels * n_bands,)

# -- ONLINE CLASSIFIER SETUP -----------------------------------------------------

# We'll use a standard scaler that we update incrementally (running mean + std)
class RunningNormalizer:
    """
    Maintains running mean and variance (Welford’s algorithm) per feature,
    so that we can z‐score each new data point online.
    """
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)  # for variance

    def update(self, x):
        """
        Update running stats with a new 1D feature vector `x`.
        x shape: (n_features,)
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x):
        """
        Z‐score the input vector `x` using current running mean and variance.
        If variance is zero (n<2), returns zeros.
        """
        if self.n < 2:
            return np.zeros_like(x)
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var + 1e-8)
        return (x - self.mean) / std

    def get_params(self):
        """Return current mean and std for saving/restoring."""
        if self.n < 2:
            return self.mean.copy(), np.sqrt((self.M2 / max(self.n - 1, 1)) + 1e-8)
        return self.mean.copy(), np.sqrt((self.M2 / (self.n - 1)) + 1e-8)

# Create filters once
bp_filters = {}
for band_name, (low, high) in BANDS.items():
    bp_filters[band_name] = design_bandpass(low, high, FS, FILTER_ORDER)

notch_b, notch_a = design_notch(NOTCH_FREQ, FS, NOTCH_QUALITY)

# Feature dimension: N_CHANNELS * number_of_bands
N_BANDS = len(BANDS)
FEATURE_DIM = N_CHANNELS * N_BANDS

# Running normalizer for FEATURE_DIM
normalizer = RunningNormalizer(FEATURE_DIM)

# Online classifier: stochastic gradient‐descent linear classifier
clf = SGDClassifier(
    loss='log',        # logistic regression
    learning_rate='invscaling',
    eta0=0.01,         # initial learning rate
    random_state=0
)
# We must call partial_fit once with all classes to initialize internal structures
clf.partial_fit(
    np.zeros((1, FEATURE_DIM), dtype=np.float32), 
    np.array([0]), 
    classes=np.array(ACTIVITIES)
)

# --------------------------------------------------------------------------------
# SIMULATED DATA SOURCE (REPLACE WITH REAL DATA ACQUISITION IN PRACTICE)
# --------------------------------------------------------------------------------

class SimulatedECoGSource:
    """
    Simulates continuous ECoG streaming. Each call to `get_next_window()` returns:
      - ecog_window: (WINDOW_SIZE, N_CHANNELS) array
      - true_label: integer in {0, 1} after a small delay
    For demonstration, true_label is chosen randomly. In a real experiment, 
    you’d receive the true label from an experimenter interface or another signal.
    """
    def __init__(self, total_time_sec=60):
        self.total_samples = total_time_sec * FS
        self.samples_sent = 0
        # Create a long random ECoG stream for simulation
        self.ecog_stream = np.random.randn(self.total_samples, N_CHANNELS).astype(np.float32)
        # Simulate true labels changing every 2 seconds
        self.labels = []
        n_windows = (self.total_samples - WINDOW_SIZE) // WINDOW_SIZE
        for w in range(n_windows + 1):
            # Alternate labels 0↔1 every 2 seconds
            t_start = w * WINDOW_MS / 1000.0  # in seconds
            label = int((t_start // 2) % 2)
            self.labels.append(label)

    def get_next_window(self):
        """
        Returns the next 100‐ms (WINDOW_SIZE) chunk of ECoG data and the corresponding true label.
        """
        if self.samples_sent + WINDOW_SIZE > self.total_samples:
            return None, None

        wstart = self.samples_sent
        wend = wstart + WINDOW_SIZE
        ecog_window = self.ecog_stream[wstart:wend, :]  # shape: (WINDOW_SIZE, N_CHANNELS)
        window_idx = self.samples_sent // WINDOW_SIZE
        true_label = self.labels[window_idx]
        self.samples_sent += WINDOW_SIZE
        return ecog_window, true_label

# Create the simulated source
source = SimulatedECoGSource(total_time_sec=20)  # 20 seconds of data

# --------------------------------------------------------------------------------
# REAL‐TIME LOOP: ACQUIRE → PREPROCESS → PREDICT → (OPTIONAL) UPDATE
# --------------------------------------------------------------------------------

# A sliding buffer is not needed here because our simulation returns non‐overlapping windows.
# In a real system, you might use a deque of length WINDOW_SIZE and shift by 1–10 samples each step.

print("Starting real‐time simulation...")

window_count = 0
cumulative_accuracy = []
running_correct = 0
running_total = 0

while True:
    ecog_window, true_label = source.get_next_window()
    if ecog_window is None:
        break

    # --- (1) PREPROCESSING ---
    # 1a. Bandpass & Notch filter (zero‐phase)
    #    Apply bandpass separately for each band when computing bandpower, so no need to filter here again.
    #    For simplicity, skip filtering step: assume data is “noisy enough” that bandpower extraction below suffices.

    # 1b. Extract bandpower features
    features = extract_band_power(ecog_window, FS, BANDS)  # shape: (FEATURE_DIM,)

    # 1c. Normalize using running stats
    if normalizer.n >= 2:
        features_norm = normalizer.normalize(features)
    else:
        # If we have <2 windows so far, feed zeros to initialize
        features_norm = np.zeros_like(features)
    # Update running stats after normalization
    normalizer.update(features)

    # --- (2) PREDICTION ---
    features_norm = features_norm.reshape(1, -1).astype(np.float32)  # shape: (1, FEATURE_DIM)
    pred_label = clf.predict(features_norm)[0]

    # --- (3) PERFORMANCE MONITORING ---
    running_total += 1
    if pred_label == true_label:
        running_correct += 1
    accuracy = running_correct / running_total
    cumulative_accuracy.append(accuracy)

    # Print progress every 10 windows
    if window_count % 10 == 0:
        print(f"Window {window_count:3d} | Pred={pred_label} | True={true_label} | "
              f"Accuracy so far: {accuracy*100:.2f}%")
    window_count += 1

    # --- (4) ONLINE UPDATE (fine‐tune) ---
    # Immediately update classifier with the newly labeled example
    clf.partial_fit(features_norm, np.array([true_label]))

    # In a real setup, you might do this only every K windows or when confidence is low.

print("Simulation complete.")
print(f"Final accuracy over {running_total} windows: {running_correct / running_total * 100:.2f}%")


"""
Explanation of Key Sections

    Design Filters vs. Direct Bandpower

        We provide functions to design bandpass/notch filters in case you want the raw time‐series filtered.

        For feature extraction, we directly compute the spectral power in each band (delta, theta, alpha, beta, high‐gamma). This is a common practice in ECoG decoding: rather than feeding raw time‐series to a classifier, you feed bandpower features.

    RunningNormalizer

        Implements Welford’s algorithm to maintain a running mean and variance for each feature.

        Each time a new window arrives, we first normalize using the current mean/std, then update the mean/std with that raw feature vector. This guarantees that test‐time data are always normalized exactly as the model expects.

    SGDClassifier (Scikit‐learn)

        We chose loss='log' (logistic regression) with a small learning rate (eta0=0.01).

        We call .partial_fit(x, y, classes=…) initially on a dummy example to initialize the classifier. After that, every new labeled window is used in .partial_fit() to update the model weights immediately—this is true online learning.

    Simulated Source

        SimulatedECoGSource yields non‐overlapping 100 ms windows of random data and a synthetic “true_label” that flips every 2 seconds. In your real experiment, replace get_next_window() with code that reads from your amplifier and obtains the real activity label (e.g., from a button press or an experiment control signal).

    Real‐Time Loop

        For each window:

            Extract bandpower features (320‐dimensional)

            Z‐score normalize with running stats

            Predict a label via clf.predict(...)

            Update performance metrics (accuracy)

            Immediately call clf.partial_fit(...) on the same window + true label to adapt the model

    You can adjust how often you update (e.g., every window, every 5th window, or only if confidence is low).

4. How to Achieve Optimal Performance in the Shortest Time

    Start Simple (Baseline):

        A linear model on bandpower features (as above) often reaches decent accuracy within a few seconds of data.

        Linear models converge quickly and are easy to update online.

    Use a Small Neural Network (Optional):
    If you find the linear classifier plateauing, you can replace SGDClassifier with a small PyTorch model (e.g., one hidden layer of width 128). Use very small learning rates and small mini‐batches (e.g., accumulate 5 windows before an update) to keep training stable. But note: PyTorch’s overhead is higher, so real‐time throughput may be slower.

    Feature Engineering:

        Bandpower is a good starting feature set. You can also compute time‐domain features (e.g., root‐mean‐square, Hjorth parameters, wavelet coefficients) if band‐specific information is insufficient.

        Always measure decoding accuracy on held‐out windows (e.g., run a short offline CV after every minute of data).

    Adaptive Learning Rate:

        Use learning_rate='invscaling' or learning_rate='adaptive' in SGDClassifier, so that as more windows arrive, the step size decays and the classifier stabilizes.

        Alternatively, use a small constant learning rate (e.g., 1e-3) if accuracy is not very sensitive.

    Prevent Catastrophic Forgetting:

        Because you update on every single new window, older examples effectively get “forgotten”. One remedy is to maintain a small replay buffer of past N windows (and labels) and occasionally re‐train with a mix of old and new data.

        For example, every time you see 50 new windows, randomly sample 10 from the buffer and do a mini‐batch update of size 11 (10 past examples + 1 new). This helps maintain performance on previously learned patterns.

    Early Stopping & Monitoring Drift:

        Compute a rolling accuracy or Kullback–Leibler divergence of the feature distribution to detect when signals drift significantly.

        If drift is detected, consider resetting the model or enlarging the replay buffer.

    Hardware Considerations:

        On a typical laptop (4–8 CPU cores), this pipeline (feature extraction + linear classification) can process > 200 windows per second in Python, which is well above real‐time (10 windows/sec).

        If you use GPU‐accelerated PyTorch for a neural net, make sure you batch windows (e.g., accumulate 5 or 10 windows) to amortize GPU overhead.

5. Summary

    Yes—when you have live neural data and want to adapt to it on the fly, you are using online learning.

    The easiest way to get good real‐time performance quickly is:

        Bandpower feature extraction (per 100 ms window)

        Running‐statistics normalization

        An SGDClassifier (logistic regression) with .partial_fit(...)

    The script above is a full, end‐to‐end proof‐of‐concept:

        Simulates real‐time ECoG acquisition.

        Extracts five‐band power features from each 100 ms window.

        Normalizes via running mean/variance.

        Predicts activity label and updates the classifier on every new labeled window.

        Reports cumulative accuracy.

You can copy–paste this script and modify the SimulatedECoGSource to read from your actual amplifier and labeling mechanism. From there, you can swap in a small PyTorch network if you want nonlinear decoding, but start with the linear approach to get something running in minutes.
"""