#%%
import argparse
import torch
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import librosa
import noisereduce
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import align, butter_highpass_filter
import time
import math
#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--event_path", type=str, help="File name", default="EventRecordings/abespeech_chipbag.npy",
)
parser.add_argument(
    "--gt_path", type=str, help="File name", default="GroundTruth/abespeech_gt.mp3",
)
parser.add_argument(
    "--out_path", type=str, help="File name", default="Output/abespeech_chipbag_offline_hat.wav",
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
t0 = time.time()

events["x"] -= events["x"].min()
events["y"] -= events["y"].min()
good_inds = (events["x"] <= 100) & (events["y"] <= 100)
events = events[good_inds]
events["t"] = events["t"] - events["t"].min()

f = int(2.5e4)

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

hist_norm = (
    (process - process.min()) * 255 / (process.max() - process.min())
).astype(np.uint8)

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    with tqdm(total=len(hist_norm) - 1) as pbar:
        def calculate_flow(i):
            # Calculate optical flow for a pair of frames
            flow = cv2.calcOpticalFlowFarneback(
                hist_norm[i], hist_norm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0 #default params from opencv
            )
            # spatially integrate flow with importance weighting (based on spatial event counts)
            w = np.abs(process[i]) + np.abs(process[i + 1])
            w = w / w.sum().clip(1e-6)
            pbar.update(1)
            return (w[..., None]*flow).sum((0, 1))
        results = [i for i in executor.map(calculate_flow, range(len(process) - 1))]
flow_summed = np.stack(results, axis=0)
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

t = time.time() - t0
print("Time taken:", t, "Recording time:", events["t"].max() * 1e-6)

#Now evaluate PESQ and STOI
out_der_sr = align(out_der_sr, gt)

out_der_16khz = librosa.resample(out_der_sr, orig_sr=sr, target_sr=16000)
gt_16khz = librosa.resample(gt, orig_sr=sr, target_sr=16000)
psq, sto = pesq(16000, gt_16khz, out_der_16khz, 'wb'), stoi(gt, out_der_sr, sr, extended=False)
print("pesq is", psq, "stoi is", sto)

out_path = out_path[:-4] + f"_pesq{psq:.2f}_stoi{sto:.2f}.wav"
sf.write(out_path, out_der_sr, sr)
#%%