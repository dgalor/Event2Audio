#%%
import numpy as np
import argparse
import os
from scipy.signal import correlate, correlation_lags
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
from scipy.signal import butter, lfilter
import soundfile as sf
import librosa
from sklearn.decomposition import PCA
import time
import math
import numpy as np
from metavision_sdk_cv import TripletMatchingFlowAlgorithm, SparseOpticalFlowAlgorithm, TimeGradientFlowAlgorithm, PlaneFittingFlowAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from tqdm import trange
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import librosa
import time

# path = "data/abspeech_chipbag.npy"
# # path = "abe_speech_bag.npy"
# gt_path = "data/gt/abe_speech.mp3"
# out_path = path.replace(".npy", "_online_hat.wav")

#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, help="File name", default="data/abspeech_chipbag.npy",
)
parser.add_argument(
    "--gt_path", type=str, help="File name", default="data/gt/abe_speech.mp3",
)
parser.add_argument(
    "--out_path", type=str, help="File name", default="data/hat/abspeech_chipbag_online_hat.wav",
)
args = parser.parse_args()

path = args.data_path
gt_path = args.gt_path
out_path = args.out_path
#%%
gt, sr = librosa.load(gt_path, sr=16000)

if os.path.exists(out_path):
    sr_, out_der_sr = wavfile.read(out_path)
    assert sr_ == 16000
    print("File {out_path} already exists: pesq is", pesq(sr, gt, out_der_sr, 'wb'), "stoi is", stoi(gt, out_der_sr, sr, extended=False))
    exit(0)

events = np.load(path)
t1 = time.time()

events["x"] -= events["x"].min()
events["y"] -= events["y"].min()
good_inds = (events["x"] <= 100) & (events["y"] <= 100)
events = events[good_inds]
events["t"] = events["t"] - events["t"].min()

f = 1e5

nbins = int(math.ceil(events["t"].max()*1e-6 * f))

relative_t = (events["t"] / 1e6) * f
coord_t = np.round(relative_t).astype(np.uint32)
coord_p = events["p"].astype(np.uint32)

w = events["x"].max() + 1
h = events["y"].max() + 1
#%%
# alg_denoise = TrailFilterAlgorithm(w, h, 100)
# alg_denoise = SpatioTemporalContrastAlgorithm(w, h, 10, False)
# alg_denoise = ActivityNoiseFilterAlgorithm(w, h, 1000)
alg = TimeGradientFlowAlgorithm(w, h, radius = 7, min_flow_mag = 1.0, bit_cut = 0)
# buffer = alg_denoise.get_empty_output_buffer()
# alg_denoise.process_events(events, buffer)
# events = buffer.numpy().copy()
buffer = alg.get_empty_output_buffer()
alg.process_events(events, buffer)
out = buffer.numpy().copy()
#%%
tv, vx, vy = out["t"], out["vx"], out["vy"]
coord_tv = np.round((tv/1e6)*f).astype(np.uint32)
# counts = coo_matrix((np.ones_like(coord_tv), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat_x = coo_matrix((vx, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat_y = coo_matrix((vy, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# mat_x = mat_x / counts.clip(1, None)
# mn, std = np.abs(mat_x).mean(), np.abs(mat_x).std()
# mat_x = np.clip(np.abs(mat_x), mn-5*std, mn+5*std) * np.sign(mat_x)
# mat_y = mat_y / counts.clip(1, None)
# mn, std = np.abs(mat_y).mean(), np.abs(mat_y).std()
# mat_y = np.clip(np.abs(mat_y), mn-5*std, mn+5*std) * np.sign(mat_y)
# print("Mid Time taken:", time.time()-t1, "recording time:", events["t"].max()*1e-6)

#%%
def align(x_hat, x):
    assert (x_hat.ndim == 1) and (x.ndim == 1)
    if len(x) > len(x_hat):
        x_hat = np.pad(x_hat, (0, len(x)-len(x_hat)))
    correlation = correlate(x, x_hat, mode='full')
    lags = correlation_lags(len(x), len(x_hat), mode='full')
    peak_ind = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_ind]
    #now we have the peak lag, we can shift, crop, and pad x_hat to align to x
    x_hat = np.roll(x_hat, peak_lag)
    #set rolled values to 0
    if peak_lag > 0:
        x_hat[:peak_lag] = 0
    elif peak_lag < 0:
        x_hat[peak_lag:] = 0
    if correlation[peak_ind] < 0:
        print("Negative correlation, flipping sign")
        x_hat = -x_hat
    print("Peak lag:", peak_lag)
    return x_hat[:len(x)]
s = np.stack((mat_x, mat_y), axis=0)
i_highest = np.argmax(np.linalg.norm(s, axis=1))
i_lowest = np.argmin(np.linalg.norm(s, axis=1))
assert i_highest != i_lowest
s[i_lowest] = align(s[i_lowest], s[i_highest])
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
#%%
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
out_der_ = s.mean(0) #pca.fit_transform(s.T)[:,0]
# out_der_ = butter_lowpass_filter(out_der_, 2000, f)
out_der_ = np.cumsum(out_der_, axis=-1)
out_der_ = butter_highpass_filter(out_der_, 100, f)
out_der_ = out_der_ / np.abs(out_der_).max()
#%%
t = time.time() - t1

print("Time taken:", t, "recording time:", events["t"].max()*1e-6)
#now resample out_der to gt
out_der_sr = librosa.resample(out_der_, orig_sr=f, target_sr=sr)

out_der_sr = align(out_der_sr, gt)
psq, sto = pesq(sr, gt, out_der_sr, 'wb'), stoi(gt, out_der_sr, sr, extended=False)
print("pesq is", psq, "stoi is", sto)

out_path = out_path[:-4] + f"_pesq{psq:.2f}_stoi{sto:.2f}_{t:.1f}s.wav"
sf.write(out_path, out_der_sr, sr)
#%%