#%%
import numpy as np
import argparse
import os
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import librosa
import noisereduce
import time
import math
from metavision_sdk_cv import TimeGradientFlowAlgorithm
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from utils import align, butter_highpass_filter
#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--event_path", type=str, help="File name", default="EventRecordings/abespeech_chipbag.npy",
)
parser.add_argument(
    "--gt_path", type=str, help="File name", default="GroundTruth/abespeech_gt.mp3",
)
parser.add_argument(
    "--out_path", type=str, help="File name", default="Output/abespeech_chipbag_online_hat.wav",
)
args = parser.parse_args()

path = args.event_path
gt_path = args.gt_path
out_path = args.out_path
#%%
sr = 44100
gt, _ = librosa.load(gt_path, sr=sr)

if os.path.exists(out_path):
    sr_, out_der_sr = wavfile.read(out_path)
    assert sr_ == sr
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
alg = TimeGradientFlowAlgorithm(w, h, radius = 7, min_flow_mag = 1.0, bit_cut = 0)
buffer = alg.get_empty_output_buffer()
alg.process_events(events, buffer)
out = buffer.numpy().copy()
#%%
tv, vx, vy = out["t"], out["vx"], out["vy"]
coord_tv = np.round((tv/1e6)*f).astype(np.uint32)
mat_x = coo_matrix((vx, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat_y = coo_matrix((vy, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
#%%

s = np.stack((mat_x, mat_y), axis=0)
i_highest = np.argmax(np.linalg.norm(s, axis=1))
i_lowest = np.argmin(np.linalg.norm(s, axis=1))
assert i_highest != i_lowest
s[i_lowest] = align(s[i_lowest], s[i_highest])
#%%
out_der_ = s.mean(0)
out_der_ = np.cumsum(out_der_, axis=-1)
out_der_ = butter_highpass_filter(out_der_, 100, f)
out_der_ = out_der_ / np.abs(out_der_).max()
#%%
#now resample out_der to gt
out_der_sr = librosa.resample(out_der_, orig_sr=f, target_sr=sr)
out_der_sr = noisereduce.reduce_noise(
    y=out_der_sr, 
    sr=sr, 
    prop_decrease=0.8, 
    freq_mask_smooth_hz=50, 
    time_mask_smooth_ms=100, 
    n_fft=4096,
)

t = time.time() - t1
print("Time taken:", t, "recording time:", events["t"].max()*1e-6)
#%%
#Now evaluate PESQ and STOI
out_der_sr = align(out_der_sr, gt)
out_der_16khz = librosa.resample(out_der_sr, orig_sr=sr, target_sr=16000)
gt_16khz = librosa.resample(gt, orig_sr=sr, target_sr=16000)
psq, sto = pesq(16000, gt_16khz, out_der_16khz, 'wb'), stoi(gt, out_der_sr, sr, extended=False)
print("pesq is", psq, "stoi is", sto)

out_path = out_path[:-4] + f"_pesq{psq:.2f}_stoi{sto:.2f}.wav"
sf.write(out_path, out_der_sr, sr)
#%%