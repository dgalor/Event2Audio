#%%
import numpy as np
import optical_flow
import argparse
import torch
import os
import time

events_raw = np.load("abe_speech_bag.npy")

s, e, f = 2, 12, int(1 / 4e-5)
cond = (events_raw["t"] >= s * 1e6) & (events_raw["t"] < e * 1e6)
events = events_raw[cond]

nbins = int(round((e - s) * f))
relative_t = (events["t"] / 1e6 - s) * f
coord_t = np.round(relative_t).astype(np.uint32)
coord_p = events["p"].astype(np.uint32)
val_x = events["x"] - events["x"].min()
val_y = events["y"] - events["y"].min()

coords = torch.from_numpy(
    np.stack((coord_t, coord_p, val_y, val_x), axis=-1).astype(np.int32)
)
#%%
%%time

frames_0 = torch.sparse_coo_tensor(
    coords.T,
    torch.ones(len(coords), dtype=torch.int8),
    (nbins, 2, 101, 101),
).to_dense()

process = frames_0.float()
process = process / process.mean(
    (0, 2, 3), keepdim=True
)  # normalize based on polarities
process = (
    process[:, 1] - process[:, 0]
)  # represent as positive event count - negative event count

flows, _, _ = optical_flow.compute_optical_flow_parallel(process.numpy())
flows = np.stack(flows, axis=0)
print(
    f"finish computing optical flow"
)
#%%
%%time
abs_process = np.abs(process)
w = ((abs_process[1:] + abs_process[:-1])/2)
del abs_process
flow_summed = (w[..., None]*flows).sum((1, 2)) / w.sum((1, 2)).clip(1e-6, None)[..., None]
flow_summed = np.pad(flow_summed, ((0, 1), (0, 0)))
#%%
import matplotlib.pyplot as plt
plt.plot(flow_summed)
#%%
def align(x_hat, x):
    from scipy.signal import correlate
    peak_lag = np.inf
    while peak_lag != 0:
        correlation = correlate(x_hat, x, mode='full')
        lags = np.arange(-len(x_hat) + 1, len(x))
        #abs peak
        peak_ind = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_ind]
        if peak_lag < 0:
            x_hat = np.pad(x_hat, (-peak_lag, 0), constant_values=0)#out_der_sr[0])
            x_hat = x_hat[:len(x)]
        else:
            x_hat = np.pad(x_hat, (0, peak_lag), constant_values=0)#out_der_sr[-1])
            x_hat = x_hat[-len(x):]
        if correlation[peak_ind] < 0:
            print("Negative correlation, flipping sign")
            x_hat = -x_hat
        print("Peak lag:", peak_lag)
    x_hat = np.pad(x_hat, (0, len(x)-len(x_hat)), constant_values=0)#out_der_sr[-1])
    return x_hat
import soundfile as sf
import librosa
from sklearn.decomposition import PCA
s = np.pad(flow_summed, ((0, 1), (0, 0))).T
gt = librosa.load("abe_speech_gt.mp3", sr=16000)[0]

s[0] = align(s[0], s[1])
pca = PCA(n_components=1)
out_der = pca.fit_transform(s.T)[:,0]
#%%
from scipy.signal import butter, lfilter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y
def mel(out, f=f):
    n_mels = 100
    fmax = 4096
    fmin = 100
    sr = f
    y = out
    y = y / np.abs(y).max()
    fig, ax = plt.subplots(figsize=(12,5))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin, n_fft=8048)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, fmax=fmax,fmin=fmin,sr=sr)
    plt.colorbar(img,ax=ax)
    plt.title('Speech Recovery (Ours)')
    plt.show()
from copy import deepcopy
out_der_ = out_der
# out_der_ = butter_lowpass_filter(out_der_, 4096, f)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = np.cumsum(out_der_, axis=-1)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = butter_highpass_filter(out_der_, 100, f)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = out_der_ / np.abs(out_der_).max()
#%%
#now resample out_der to gt
out_der_sr = librosa.resample(out_der_, orig_sr=f, target_sr=16000)
import noisereduce
# out_der_sr = noisereduce.reduce_noise(y=out_der_sr, sr=16000)
out_der_sr = align(out_der_sr, gt)
from pesq import pesq
from pystoi import stoi
print("pesq is", pesq(16000, gt, out_der_sr, 'wb'), "stoi is", stoi(gt, out_der_sr, 16000, extended=False))

sf.write("abe_speech_bag_hat_new.wav", out_der_sr, 16000)
#%%