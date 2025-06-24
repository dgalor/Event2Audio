from scipy.signal import butter, lfilter, correlate, correlation_lags
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

def offline_optical_flow(hist_frames):
    """
    Compute optical flow between consecutive histogram frames using parallel processing.

    Parameters
    ----------
    hist_frames : list or ndarray
        List/array of histogram frames (time, height, width)
    """
    # hist_norm = (hist_frames * 255 / np.max(hist_frames)).astype(np.uint8)
    hist_norm = (
        (hist_frames - hist_frames.min()) * 255 / (hist_frames.max() - hist_frames.min())
    ).astype(np.uint8)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(hist_frames) - 1) as pbar:
            def calculate_flow(i):
                # Calculate optical flow for a pair of frames
                flow = cv2.calcOpticalFlowFarneback(
                    hist_norm[i], hist_norm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                w = np.abs(hist_frames[i]) + np.abs(hist_frames[i + 1])
                w = w / w.sum().clip(1e-6)
                pbar.update(1)
                return (w[..., None]*flow).sum((0, 1))
            results = [i for i in executor.map(calculate_flow, range(len(hist_frames) - 1))]
    return np.stack(results, axis=0)

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