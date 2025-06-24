#%%
import argparse
import torch
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import soundfile as sf
from tqdm import trange
import librosa
import noisereduce
import cv2
import numpy as np
from tqdm import tqdm
from utils import align, butter_highpass_filter, offline_optical_flow
import time
import math
#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--event_path", type=str, help="File name", default="EventRecordings/abespeech_chipbag.npy",
)
parser.add_argument(
    "--gt_path", type=str, help="File name", default="GroundTruth/abespeech_gt.wav",
)
parser.add_argument(
    "--out_path", type=str, help="File name", default="Output/abespeech_chipbag_hat_offline.wav",
)
args = parser.parse_args()

path = args.event_path
gt_path = args.gt_path
out_path = args.out_path
#%%

if os.path.exists(out_path):
    sr, out_der_sr = wavfile.read(out_path)
    gt, _ = librosa.load(gt_path, sr=sr)
    print("File {out_path} already exists: pesq is", pesq(sr, gt, out_der_sr, 'wb'), "stoi is", stoi(gt, out_der_sr, sr, extended=False))
    exit(0)
    
f = int(2.5e4)
print("Loaded ground truth from:", gt_path)
gt, _ = librosa.load(gt_path, sr=f)
events = np.load(path)
print("Loaded events from", path)
t1 = time.time()

events["x"] -= events["x"].min()
events["y"] -= events["y"].min()
good_inds = (events["x"] <= 100) & (events["y"] <= 100)
events = events[good_inds]
events["t"] = events["t"] - events["t"].min()

nbins = int(math.ceil(events["t"].max()*1e-6 * f))

relative_t = (events["t"] / 1e6) * f
coord_t = np.round(relative_t).astype(np.uint32)
coord_p = events["p"].astype(np.uint32)

w = events["x"].max() + 1
h = events["y"].max() + 1
coords = torch.from_numpy(
    np.stack((coord_t, coord_p, events["y"], events["x"]), axis=-1).astype(np.int32)
)
#%%
process = torch.sparse_coo_tensor(
    coords.T,
    torch.ones(len(coords), dtype=torch.int8),
    (nbins, 2, h, w),
).to_dense().float()

process = process / process.mean(
    (0, 2, 3), keepdim=True
)  # normalize based on polarities since there is asymmetrical sensitivity
process = (
    process[:, 1] - process[:, 0]
).numpy()  # represent as positive (normed) event count - negative (normed) event count

flow_summed = offline_optical_flow(process)
#%%

s = np.pad(flow_summed, ((0, 1), (0, 0))).T
i_highest = np.argmax(np.linalg.norm(s, axis=1))
i_lowest = np.argmin(np.linalg.norm(s, axis=1))

s[i_lowest] = align(s[i_lowest], s[i_highest])
#%%
out_der_ = s.mean(0)
out_der_ = np.cumsum(out_der_, axis=-1)
out_der_ = butter_highpass_filter(out_der_, 100, f)
out_der_ = out_der_ / np.abs(out_der_).max()
#%%
#align the output derivative to the ground truth
out_signal_f = align(out_der_, gt)

#resample to standard 44100Hz
out_signal_sr = librosa.resample(out_signal_f, orig_sr=f, target_sr=44100)

winsize = 0.1 # window size in seconds for noise reduction
#denoise the output using spectral gating
out_signal_sr = noisereduce.reduce_noise(
    y=out_signal_sr, 
    sr=44100, 
    prop_decrease=0.8, 
    freq_mask_smooth_hz=50, 
    time_mask_smooth_ms=100, 
    win_length=int(44100 * winsize),
    n_fft=int(44100 * winsize),  # Adjust based on window size
)
t = time.time() - t1
print("Time taken:", t, "recording time:", events["t"].max()*1e-6)
#%%
#now resample signals to 16khz for compatibility with evals
out_signal_16khz = librosa.resample(out_signal_sr, orig_sr=44100, target_sr=16000)
gt_16khz, _ = librosa.load(gt_path, sr=16000)

out_signal_16khz = align(out_signal_16khz, gt_16khz) # realign in case 1 pixel mismatch due to resampling
#Now evaluate PESQ and STOI
psq, sto = pesq(16000, gt_16khz, out_signal_16khz, 'wb'), stoi(gt_16khz, out_signal_16khz, 16000, extended=False)
print("pesq is", psq, "stoi is", sto)

out_path_44khz = out_path[:-4] + f"_44khz.wav"
sf.write(out_path_44khz, out_signal_sr, 44100)

out_path = out_path[:-4] + f"_pesq{round(psq, 2):.2f}_stoi{round(sto, 2):.2f}.wav"
sf.write(out_path, out_signal_16khz, 16000)
sf.write(gt_path[:-4] + "_16khz.wav", gt_16khz, 16000)
#%%