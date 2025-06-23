from scipy.signal import butter, lfilter, correlate, correlation_lags
import numpy as np

# Function to apply a Butterworth high-pass filter
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist,
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

# align x_hat to x using cross-correlation
def align(x_hat, x):
    assert (x_hat.ndim == 1) and (x.ndim == 1)
    if len(x) > len(x_hat):
        x_hat = np.pad(x_hat, (0, len(x)-len(x_hat)))
    correlation = correlate(x, x_hat, mode='full')
    lags = correlation_lags(len(x), len(x_hat), mode='full')
    peak_ind = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_ind]
    x_hat = np.roll(x_hat, peak_lag)
    if peak_lag > 0:
        x_hat[:peak_lag] = 0
    elif peak_lag < 0:
        x_hat[peak_lag:] = 0
    if correlation[peak_ind] < 0:
        x_hat = -x_hat
    return x_hat[:len(x)]